import os
from pathlib import Path

import torch
from torch import nn
from typing import Any

def print0(*args: Any, **kwargs: Any) -> None:
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def is_checkpoint_step(step: int) -> bool:
    # step & (step + 1) == 0 iff step + 1 is a power of 2. Therefore, the following
    # expression will be true iff step is one less than a power of two between 0
    # and 1000 or if step + 1 is a multiple of 1000.
    return (0 <= step < 1000 and (step & (step + 1)) == 0) or (step + 1) % 1000 == 0


def save_model_and_config(save_dir: Path, model: nn.Module, step: int) -> None:
    """Save the model to disk. Also save the config file if it doesn't exist.

    Args:
        save_dir: The directory to save the model and config to.
        model: The model to save.
        step: The current step (used in the model filename).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    model_file = save_dir / f"model_step_{step}.pt"
    torch.save(model.state_dict(), model_file)
    print0(f"Saved model to {model_file}")
