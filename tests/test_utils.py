from simple_stories_train.utils import is_checkpoint_step


def test_is_checkpoint_step() -> None:
    checkpoint_steps = [1, 2, 4, 8, 512, 1000, 2000, 10000]
    non_checkpoint_steps = [3, 5, 1024, 2048]
    for step in checkpoint_steps:
        assert is_checkpoint_step(step) is True
    for step in non_checkpoint_steps:
        assert is_checkpoint_step(step) is False
