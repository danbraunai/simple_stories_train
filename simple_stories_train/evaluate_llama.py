"""
Simple Story Generator Evaluation Script

This script evaluates language models on story generation capabilities using the SimpleStories dataset.
It uses the inspect-ai framework to load a pretrained model, generate stories from prompts, and
evaluate the quality of the generated stories based on originality, coherence, grammar, and overall quality.

Arguments:
    model_path (str): Path to the pretrained model weights
    model_config (str): Name of the model configuration in MODEL_CONFIGS
    tokenizer_path (str): Path to the tokenizer file
    tokenizer_eos_token (str): End-of-sequence token for the tokenizer. Defaults to '[EOS]'
    device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

Example:
    python evaluate_stories.py --model_path="./models/model.pt"
                              --model_config="33M"
                              --tokenizer_path="./tokenizer.json"
                              --tokenizer_eos_token="</s>"
                              --device="cuda"

Majority of the script is taken from https://github.com/danbraunai/simple_stories_train/blob/evaluation/simple_stories_train/evaluate.py
TODO: Giving '[EOS]' as tokenizer_eos_token on cli causes error as python sees it as a list rather
than string. Gotta handle that better
"""

import os
import re
from collections.abc import Callable

import fire
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Dataset, Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    get_model,
    modelapi,
)
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState, generate
from inspect_ai.tool import ToolChoice, ToolInfo
from models.llama import Llama
from models.model_configs import MODEL_CONFIGS
from pydantic import BaseModel, ConfigDict
from tokenizers import Tokenizer


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    split: str
    column_name: str
    n_ctx: int = 1024
    seed: int = 42
    prompt_words: int = 18


class HuggingFaceStoriesDataset(Dataset):
    def __init__(self, config: DatasetConfig, limit: int = 1000):
        self.config = config

        # Load dataset from Hugging Face
        self.hf_dataset = load_dataset(config.name, split=config.split)

        if limit:
            self.hf_dataset = self.hf_dataset.select(range(min(limit, len(self.hf_dataset))))  # type: ignore

        # Create samples
        self.samples = []
        for idx, item in enumerate(self.hf_dataset):
            truncated_text = self._truncate_text(item[config.column_name])
            self.samples.append(Sample(input=truncated_text, id=str(idx)))

        self._shuffled = False

    def _truncate_text(self, text: str) -> str:
        """Truncate text to specified number of words."""
        words = text.split()
        if len(words) <= self.config.prompt_words:
            return text
        return " ".join(words[: self.config.prompt_words])

    def __getitem__(self, index: int) -> Sample:  # type: ignore[override]
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    def filter(self, predicate: Callable[[Sample], bool]) -> "HuggingFaceStoriesDataset":  # type: ignore[override]
        filtered_samples = [sample for sample in self.samples if predicate(sample)]
        new_dataset = HuggingFaceStoriesDataset(self.config, len(filtered_samples))
        new_dataset.samples = filtered_samples
        return new_dataset

    @property
    def location(self) -> str:
        return self.config.name

    @property
    def name(self) -> str:
        return f"huggingface_dataset_{self.config.name}"

    def shuffle(self, seed: int | None = None) -> None:
        if seed is not None:
            import random

            random.seed(seed)
            random.shuffle(self.samples)
        else:
            import random

            random.shuffle(self.samples)
        self._shuffled = True

    def shuffle_choices(self, seed: int | None = None) -> None:
        # No choices to shuffle in this dataset
        pass

    @property
    def shuffled(self) -> bool:
        return self._shuffled

    def sort(self, key: callable) -> None:  # type:ignore
        self.samples.sort(key=key)


class SimpleStoriesAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        tokenizer_path: str,
        model_config: str,
        device: str,
        tokenizer_eos_token: str = "[EOS]",
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer_eos_token = tokenizer_eos_token

        print("EOS token is ", self.tokenizer_eos_token)

        if model_config not in MODEL_CONFIGS:
            raise ValueError("model configuration not found in MODEL_CONFIGS")
        self.model = Llama.from_pretrained(model_path, MODEL_CONFIGS[model_config])

        self.model.to(device)
        self.model.eval()
        self.device = device

    async def generate(
        self,
        input: list[ChatMessage] | str,
        tools: list[ToolInfo] | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        # Handle both string and ChatMessage input
        prompt = input[0].content if isinstance(input, list) else input

        # Tokenize input
        encoding = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(encoding.ids).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        eos_token_id = self.tokenizer.token_to_id(self.tokenizer_eos_token)

        # Set up generation config
        gen_config = config or self.config

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                idx=input_ids,
                max_new_tokens=gen_config.max_tokens if gen_config.max_tokens else 100,
                temperature=gen_config.temperature if gen_config.temperature else 0.7,
                top_k=40,
                eos_token_id=eos_token_id,
            )

        # Decode output
        output_text = self.tokenizer.decode(output_ids[0].tolist())

        # Create model output
        choice = ChatCompletionChoice(
            message=ChatMessageAssistant(content=output_text),
            logprobs=None,
        )

        return ModelOutput(
            model=self.model_name,
            choices=[choice],
            usage=ModelUsage(
                input_tokens=len(encoding.ids),
                output_tokens=len(output_ids[0]) - len(encoding.ids),
                total_tokens=len(output_ids[0]),
            ),
        )


@modelapi(name="simple_stories")
def simple_stories():
    return SimpleStoriesAPI


@scorer(
    metrics={
        "originality": [mean()],
        "coherence": [mean()],
        "grammar": [mean()],
        "quality": [mean()],
    }
)
def story_quality_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        load_dotenv()
        api_key = os.getenv("API_KEY")
        judge_model = get_model("openai/gpt-4", api_key=api_key)

        generated_story = state.output.completion

        answer_match = re.search(r"ANSWER:(.*?)(?=ANSWER:|$)", generated_story, re.DOTALL)
        story = answer_match.group(1).strip() if answer_match else generated_story

        # Craft prompt for the judge
        evaluation_prompt = f"""
Evaluate the following story based on four criteria by assigning each a score from 0 to 100:
1. **Originality**: Rate the creativity and uniqueness of the story.
2. **Coherence**: Rate the logical flow and consistency of the story.
3. **Grammar**: Rate the grammatical correctness of the story. Ignore spacing and capitalization.
4. **Quality**: Rate the overall quality of the story.
You should also provide a short explanation for your judgment.

**Story to evaluate:**
{story}

Please provide your assessment in the following format, ensuring each score is an integer between 0 and 100:
{{"EXPLANATION": "The dialogue is coherent, but the phrasing is slightly off.","ORIGINALITY": 0, "COHERENCE": 0, "GRAMMAR": 0, "QUALITY": 0}}
"""

        # Get evaluation from judge model
        result = await judge_model.generate(
            evaluation_prompt, config=GenerateConfig(temperature=0.0)
        )

        # Use regex to find a dictionary with explanation and four scores
        dict_match = re.search(
            r'\{"EXPLANATION":\s*"([^"]+)",\s*"ORIGINALITY":\s*(\d+),\s*"COHERENCE":\s*(\d+),\s*"GRAMMAR":\s*(\d+),\s*"QUALITY":\s*(\d+)\}',
            result.completion,
        )

        if dict_match:
            explanation = dict_match.group(1)
            scores = {
                "originality": int(dict_match.group(2)),
                "coherence": int(dict_match.group(3)),
                "grammar": int(dict_match.group(4)),
                "quality": int(dict_match.group(5)),
            }
        else:
            explanation = ""
            scores = {"originality": 0, "coherence": 0, "grammar": 0, "quality": 0}

        scores = {k: max(0, min(v, 100)) for k, v in scores.items()}

        return Score(
            value=scores,  # type:ignore
            answer=story,
            explanation=explanation,
        )

    return score


@task
def evaluate_story_generation():
    """Task definition for evaluating story generation capabilities."""
    # Define the dataset configuration
    dataset_config = DatasetConfig(
        name="lennart-finke/SimpleStories",  # Use an actual HF dataset ID or local path
        split="test",
        column_name="story",
        prompt_words=18,
    )

    # Create the dataset
    dataset = HuggingFaceStoriesDataset(
        config=dataset_config,
        limit=100,  # Adjust as needed
    )

    return Task(
        dataset=dataset,
        plan=[generate()],
        scorer=story_quality_scorer(),
    )


def main(
    model_path: str,
    model_config: str,
    tokenizer_path: str,
    tokenizer_eos_token: str = "[EOS]",
    device: str = "cpu",
):
    model = get_model(
        "simple_stories/",
        model_path=model_path,
        model_config=model_config,
        tokenizer_path=tokenizer_path,
        tokenizer_eos_token=tokenizer_eos_token,
        device=device,
    )

    eval(evaluate_story_generation, model=model, limit=100, max_tokens=300)


if __name__ == "__main__":
    fire.Fire(main)
