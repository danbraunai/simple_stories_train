import os
from pathlib import Path
from typing import Any, TypeVar

import torch
import wandb
import yaml
from pydantic import BaseModel
from torch import nn

REPO_ROOT = Path(__file__).parent.parent


def print0(*args: Any, **kwargs: Any) -> None:
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def is_checkpoint_step(step: int) -> bool:
    # step & (step - 1) == 0 iff step is a power of 2. Therefore, the following
    # expression will be true iff step is a power of two between 0 and 1000
    # or step is a multiple of 1000.
    return (0 < step < 1000 and (step & (step - 1)) == 0) or step % 1000 == 0


def save_model_and_config(
    save_dir: Path,
    model: nn.Module,
    config_dict: dict[str, Any],
    step: int,
    config_filename: str = "final_config.yaml",
) -> None:
    """Save the model to disk and wandb. Also save the config file if it doesn't exist.

    Args:
        save_dir: The directory to save the model and config to.
        model: The model to save.
        step: The current step (used in the model filename).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    config_file = save_dir / config_filename
    if not config_file.exists():
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)
    model_file_name = f"model_step_{step}.pt"
    model_file = save_dir / model_file_name
    torch.save(model.state_dict(), model_file)
    print0(f"Saved model to {model_file}")
    if config_dict.get("wandb_project"):
        wandb.save(str(model_file), policy="now", base_path=save_dir)
        print0(f"Saved model to wandb: {str(model_file_name)}")


def log_metrics(step: int, metrics: dict[str, Any]) -> None:
    wandb.log(metrics, step=step)


def log_generations(step: int, generations: list[list[str]]) -> None:
    wandb.log(
        {
            "generation_tables": wandb.Table(
                data=generations,
                columns=["step", "generated text"],
            )
        },
        step=step,
    )


T = TypeVar("T", bound=BaseModel)


def load_config(config_path_or_obj: Path | str | T | None, config_model: type[T]) -> T:
    """Load the config of class `config_model`, either from YAML file, existing config object, or None.

    Args:
        config_path_or_obj: If config object, must be instance of `config_model`. If str or Path,
            this must be the path to a .yaml. If None, creates a default config.
        config_model: the class of the config that we are loading
    """
    if config_path_or_obj is None:
        return config_model()

    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(config_path_or_obj, Path), f"invalid config type {type(config_path_or_obj)}"
    assert config_path_or_obj.suffix == ".yaml", f"Config file {config_path_or_obj} must be .yaml."
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)
