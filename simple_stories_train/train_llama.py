"""
Training script. Currently only supports models with the Llama architecture.

Usage:
```
python train_llama.py [PATH_TO_YAML_CONFIG]
```
where `PATH_TO_YAML_CONFIG` contains the training config. If no path is provided, a default config
will be used.
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

import fire
import numpy as np
import tiktoken
import torch
import torch._inductor.config as torch_inductor_config
import torch.distributed as dist
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch import Tensor
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from simple_stories_train.dataloaders import DatasetConfig, create_data_loader
from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS
from simple_stories_train.utils import (
    REPO_ROOT,
    is_checkpoint_step,
    load_config,
    log_generations,
    log_metrics,
    print0,
    save_model_and_config,
)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = Field(
        None, description="WandB project name. If None, will not use WandB."
    )
    train_dataset_config: DatasetConfig = Field(
        DatasetConfig(
            name="lennart-finke/SimpleStories",
            is_tokenized=False,
            tokenizer_file_path="simple_stories_train/tokenizer/stories-3072.json",
            streaming=True,
            split="train",
            n_ctx=1024,
            seed=0,
            column_name="story",
        ),
        description="Dataset config for training",
    )
    val_dataset_config: DatasetConfig = Field(
        DatasetConfig(
            name="lennart-finke/SimpleStories",
            is_tokenized=False,
            tokenizer_file_path="simple_stories_train/tokenizer/stories-3072.json",
            streaming=True,
            split="test",
            n_ctx=1024,
            seed=0,
            column_name="story",
        ),
        description="Dataset config for validation",
    )
    output_dir: Path = Field(
        REPO_ROOT / "out", description="Directory to write logs and checkpoints"
    )
    model_name: str = Field(
        "d2",
        description=f"Name of the model to train (one of {tuple(MODEL_CONFIGS.keys())}). "
        "Currently only supports models with the Llama architecture.",
    )
    batch_size: PositiveInt = Field(4, description="Batch size")
    total_batch_size: PositiveInt = Field(
        4096, description="Number of batch_size * sequence_length before updating gradients"
    )  # TODO: Rename/reconfigure
    num_iterations: PositiveInt = Field(
        50, description="Number of gradient accumulation steps"
    )  # TODO: Allow for None and deplete the (streaming) dataset
    inference_only: bool = Field(False, description="If True, don't update gradients")
    learning_rate: PositiveFloat = Field(1e-4, description="Learning rate")
    warmup_iters: NonNegativeInt = Field(
        0, description="Number of iterations to warmup the learning rate"
    )
    learning_rate_decay_frac: PositiveFloat = Field(
        1.0, ge=0, le=1, description="Fraction of lr to decay to. 0 decays to 0, 1 doesn't decay"
    )
    weight_decay: NonNegativeFloat = Field(0.1, description="Weight decay")
    grad_clip: NonNegativeFloat | None = Field(1.0, description="Maximum gradient magnitude")
    val_loss_every: NonNegativeInt = Field(
        0, description="Every how many steps to evaluate val loss?"
    )
    val_max_steps: NonNegativeInt = Field(
        20, description="Max number of batches to use for validation"
    )
    sample_every: NonNegativeInt = Field(0, description="How often to sample from the model?")
    tensorcores: bool = Field(True, description="Use TensorCores?")
    device: str | None = Field(None, description="Device to use. If None, will autodetect.")
    compile: bool = Field(True, description="Compile the model?")
    flash_attention: bool = Field(True, description="Use FlashAttention?")
    dtype: Literal["float32", "float16", "bfloat16"] = Field("bfloat16", description="Data type")
    zero_stage: Literal[0, 1, 2, 3] = Field(
        0, description="Zero redundancy optimizer stage (0/1/2/3)"
    )

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # Check that the model name is valid
        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {self.model_name} not in {tuple(MODEL_CONFIGS.keys())}")
        return self


def main(config_path_or_obj: Path | str | Config | None = None) -> None:
    print0(f"Running pytorch {torch.__version__}")
    load_dotenv(override=True)
    config = load_config(config_path_or_obj, config_model=Config)

    B = config.batch_size
    T = config.train_dataset_config.n_ctx

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        zero_stage = config.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        # select the device
        if config.device:
            # provided explicitly by the user
            device = config.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = "cuda" if "cuda" in device else "cpu"

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert (
        config.total_batch_size % tokens_per_fwdbwd == 0
    ), f"Mismatch between batch size and tokens {config.total_batch_size} % {tokens_per_fwdbwd} != 0"
    grad_accum_steps = config.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {config.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        config.dtype
    ]
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # type: ignore
        if device_type == "cuda"
        else nullcontext()
    )

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if config.tensorcores:
        torch.set_float32_matmul_precision("high")

    # init (and write) the tokenizer
    enc: tiktoken.core.Encoding = tiktoken.get_encoding("gpt2")

    model_config = MODEL_CONFIGS[config.model_name]
    model = Llama(model_config)

    model.train()
    model.to(device)
    if config.compile:
        if hasattr(torch_inductor_config, "coordinate_descent_tuning"):
            torch_inductor_config.coordinate_descent_tuning = True  # suggested by @Chillee
        print0("compiling the model...")
        model: nn.Module = torch.compile(model)  # type: ignore[reportArgumentType]

    train_loader, _ = create_data_loader(
        dataset_config=config.train_dataset_config,
        batch_size=B,
        buffer_size=1000,
        global_seed=0,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )
    train_loader = iter(train_loader)  # Is this the right way to sample from a Pytorch DataLoader?

    val_loader, _ = create_data_loader(
        dataset_config=config.val_dataset_config,
        batch_size=B,
        buffer_size=1000,
        global_seed=0,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )
    val_loader = iter(val_loader)  # Is this the right way to sample from a Pytorch DataLoader?

    # -------------------------------------------------------------------------
    # main training loop
    if config.wandb_project is not None:
        wandb.init(project=config.wandb_project, config=config.model_dump(mode="json"))

    # here we wrap model into DDP container
    if ddp:
        model: nn.Module = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
        zero_stage=zero_stage,
    )

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it: int) -> float:
        min_lr = config.learning_rate * config.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < config.warmup_iters:
            return config.learning_rate * (it + 1) / config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_iters) / (config.num_iterations - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (config.learning_rate - min_lr)

    # create the logging directory if it does not exist
    logfile = None
    checkpoints_dir = None
    output_dir = None
    if config.output_dir:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(config.output_dir) / f"{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logfile = output_dir / "main.log"
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

        # set our checkpoints directory and save off the initilized model
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        save_model_and_config(checkpoints_dir, raw_model, config.__dict__, step=0)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    generations = []
    for step in range(1, config.num_iterations + 1):
        t0 = time.time()
        last_step = step == config.num_iterations

        # once in a while evaluate the validation dataset
        if config.val_loss_every > 0 and (step % config.val_loss_every == 0 or last_step):
            model.eval()
            val_loader_iter = iter(
                val_loader
            )  # By creating the iterator anew, we sample the same data each time
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(config.val_max_steps):
                    bat = next(val_loader_iter)
                    x = bat[-1:].view(B, T)  # inputs
                    y = bat[1:].view(B, T)  # targets
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= config.val_max_steps
            # log to wandb
            if config.wandb_project is not None:
                log_metrics(step, {"val_loss": val_loss})
            # log to console and to file
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        if (
            config.sample_every > 0 and (step % config.sample_every == 0 or last_step)
        ) and master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a
            # new sequence
            start_ids = [enc.eot_token]
            xg = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
            print0("---------------")
            print0(enc.decode(yg[0].tolist()))
            print0("---------------")
            # log to wandb
            if config.wandb_project is not None:
                generations.append([step, enc.decode(yg[0].tolist())])
            log_generations(step, generations)

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = Tensor([0.0]).to(
            device
        )  # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            bat = next(train_loader)["input_ids"].to(torch.int)
            x = bat.view(B, T)[:, :-1]  # inputs
            y = bat.view(B, T)[:, 1:]  # targets
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1  # type: ignore
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach()  # keep track of the mean loss

            # backward pass
            if not config.inference_only:
                loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = None
        if config.grad_clip is not None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1 - t0)
        norm_str = f"norm {norm:.4f}" if norm is not None else ""
        print0(
            f"step {step:4d}/{config.num_iterations} | train loss {lossf:.6f} | {norm_str} | "
            f"lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)"
        )
        # log to wandb
        if config.wandb_project is not None:
            log_metrics(
                step,
                {
                    "train_loss": lossf,
                    "lr": lr,
                },
            )
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        if checkpoints_dir is not None and is_checkpoint_step(step):
            save_model_and_config(checkpoints_dir, raw_model, config.__dict__, step=step)

        # keep track of smooth timings, last 20 iterations
        if step > 1 and step > config.num_iterations - 20:
            timings.append(t1 - t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
