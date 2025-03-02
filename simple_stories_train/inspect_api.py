"""
NOTE: To reproduce this script, as of the 01.29.2025, inspect_ai needs the manual fix described here: https://github.com/UKGovernmentBEIS/inspect_ai/issues/1103
"""

import asyncio
import copy
import functools
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import Any, Protocol, cast

import torch  # type: ignore
from huggingface_hub import PyTorchModelHubMixin
from inspect_ai._util.constants import DEFAULT_MAX_TOKENS
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    Logprobs,
    ModelAPI,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from tokenizers import Tokenizer as HF_Tokenizer  # type: ignore
from torch import Tensor  # type: ignore
from typing_extensions import override

from simple_stories_train.models.llama import Llama, LlamaConfig
from simple_stories_train.models.model_configs import MODEL_CONFIGS_DICT

HF_TOKEN = "HF_TOKEN"


class LlamaHelper(
    torch.nn.Module,
    PyTorchModelHubMixin,
):
    def __init__(self, **config: Any):
        super().__init__()
        self.llama = Llama(LlamaConfig(**config))


class SimpleStoriesAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=[HF_TOKEN],
            config=config,
        )

        # collect known model_args (then delete them so we can pass the rest on)
        def collect_model_arg(name: str) -> Any | None:
            nonlocal model_args
            value = model_args.get(name)
            if value is not None:
                model_args.pop(name)
            return value

        device = collect_model_arg("device")
        model_path = collect_model_arg("model_path")
        tokenizer_path = collect_model_arg("tokenizer_path")
        self.batch_size = collect_model_arg("batch_size")
        self.chat_template = collect_model_arg("chat_template")
        self.tokenizer_call_args = collect_model_arg("tokenizer_call_args")
        if self.tokenizer_call_args is None:
            self.tokenizer_call_args = {}

        # device
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # model
        # TODO: This line should change dynamically with the configuration of the loaded model
        llama_config = MODEL_CONFIGS_DICT["d12"]
        if model_path:
            self.model = LlamaHelper(**llama_config)
            self.model = self.model.from_pretrained(model_path)
            self.model.to(self.device)  # type: ignore
        else:
            raise ValueError("model_path is required")

        # tokenizer, should use tokenizer_path
        self.tokenizer = HF_Tokenizer.from_file("simple_stories_train/tokenizer/stories-3072.json")

        # TODO: LLMs generally don't have a pad token and we need one for batching,
        # so the following should probably be added in future
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # create chat
        # chat = self.hf_chat(input, tools)

        assert isinstance(self.tokenizer_call_args, dict)
        # prepare tokenizer
        """tokenizer = functools.partial(
            self.tokenizer,
            return_tensors="pt",
            padding=True,
            **self.tokenizer_call_args,
        )"""
        tokenizer = self.tokenizer

        # prepare generator
        kwargs: dict[str, Any] = dict()
        if config.max_tokens is not None:
            kwargs["max_new_tokens"] = config.max_tokens
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
        generator = functools.partial(self.model.llama.generate, **kwargs)  # type: ignore

        # prepare decoder
        decoder = functools.partial(
            self.tokenizer.decode,
            skip_special_tokens=True,
        )

        # generate (uses a queue to batch so we await)
        response = await batched_generate(
            GenerateInput(
                input=str(input[0].content),
                device=self.device,
                tokenizer=tokenizer,
                generator=generator,
                decoder=decoder,
                batch_size=config.max_connections or self.max_connections(),
            )
        )

        final_logprobs = None

        # construct choice
        choice = ChatCompletionChoice(
            message=ChatMessageAssistant(content=response.output, source="generate"),
            logprobs=(Logprobs(content=final_logprobs) if final_logprobs is not None else None),
        )

        # return output
        return ModelOutput(
            model=self.model_name,
            choices=[choice],
            usage=ModelUsage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
            ),
        )

    @override
    def max_tokens(self) -> int | None:
        """Default is 16, bump it up to a value suitable for evals."""
        return DEFAULT_MAX_TOKENS

    @override
    def max_connections(self) -> int:
        """Effectively the batch size."""
        return 32

    @override
    def collapse_user_messages(self) -> bool:
        return True

    def hf_chat(self, messages: list[ChatMessage], tools: list[ToolInfo]) -> str:
        # convert to hf format
        tools_list = []
        hf_messages = copy.deepcopy(messages)

        # apply chat template
        chat = self.tokenizer.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=False,
            tools=tools_list if len(tools_list) > 0 else None,
        )
        # return
        return cast(str, chat)


# return value from generate as a result of specifying return_dict_in_generate
class ModelGenerateOutput:
    sequences: Tensor
    logits: tuple[Tensor]


class Generator(Protocol):
    def __call__(self, idx: Tensor) -> Tensor: ...


class Decoder(Protocol):
    def __call__(self, sequences: Tensor) -> list[str]: ...


@dataclass
class GenerateInput:
    input: str
    device: str
    tokenizer: HF_Tokenizer
    generator: Generator
    decoder: Decoder
    batch_size: int


@dataclass
class GenerateOutput:
    output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class _QueueItem:
    input: GenerateInput
    future: asyncio.Future[GenerateOutput]
    loop: asyncio.AbstractEventLoop


batch_thread: Thread | None = None

batch_queue: "Queue[_QueueItem]" = Queue()


async def batched_generate(input: GenerateInput) -> GenerateOutput:
    # start the background thread if necessary
    global batch_thread
    if batch_thread is None:
        batch_thread = Thread(target=process_batches, daemon=True)
        batch_thread.start()

    # enqueue the job
    loop = asyncio.get_event_loop()
    future: asyncio.Future[GenerateOutput] = loop.create_future()
    batch_queue.put(_QueueItem(input=input, future=future, loop=loop))

    # await the job
    await future

    # return it
    return future.result()


def process_batches() -> None:
    while True:
        # drain the queue (wait until no new messages have shown up for 2 seconds)
        inputs: list[tuple[GenerateInput, asyncio.Future[GenerateOutput]]] = []
        while True:
            try:
                input = batch_queue.get(timeout=2)
                loop = input.loop
                inputs.append((input.input, input.future))
                if len(inputs) == input.input.batch_size:
                    # max batch size reached
                    break
            except Empty:
                # we have exhausted the queue
                break

        # see if we have any work to do
        if len(inputs) == 0:
            continue

        try:
            # capture the generator and decoder functions
            first_input = inputs[0][0]
            device = first_input.device
            tokenizer = first_input.tokenizer
            generator = first_input.generator

            # tokenize and move to device
            import threading

            lock = threading.Lock()

            with lock:
                tokenized_inputs = [tokenizer.encode(item[0].input) for item in inputs]
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(input.ids) for input in tokenized_inputs],
                    batch_first=True,
                    padding_value=-1,
                )  # TODO: Use the right padding value here
                input_ids = input_ids.to(device)

                # generate
                with torch.inference_mode():
                    generation_outputs = generator(idx=input_ids)
                    generate_ids = generation_outputs

            # decode
            generated_tokens = generate_ids[:, input_ids.size(dim=1) :]

            def process_text(s: str) -> str:
                s = s.replace(" ##", "")
                s = s.replace(" ,", ",")
                s = s.replace(" .", ".")
                s = s.replace(" ?", "?")
                s = s.replace(" !", "!")
                s = s.replace(" ;", ";")
                s = s.rsplit(".", 1)[0] + "."  # Clip output after last sentence
                return s

            outputs = [
                process_text(tokenizer.decode(generated_tokens[k, :].tolist()))
                for k in range(generated_tokens.shape[0])
            ]
            # call back futures
            for i, output in enumerate(outputs):
                future = inputs[i][1]
                input_tokens = input_ids.size(dim=1)
                output_tokens = generate_ids.size(dim=1) - input_ids.size(dim=1)

                # asyncio futures are not thread safe, so we need to pass the event loop
                # down to this point, so we can mark the future as done in a thread safe manner.
                # see: https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading
                loop.call_soon_threadsafe(  # type: ignore
                    future.set_result,
                    GenerateOutput(
                        output=output,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                )

        except Exception as ex:
            for inp in inputs:
                future = inp[1]
                loop.call_soon_threadsafe(future.set_exception, ex)  # type: ignore
