from simple_stories_train.utils import is_checkpoint_step


def test_is_checkpoint_step() -> None:
    checkpoint_steps = [1, 2, 4, 8, 512, 1000, 2000, 10000]
    non_checkpoint_steps = [3, 5, 1024, 2048]
    for step in checkpoint_steps:
        # We need to check step - 1 because the step counter starts from
        # zero and so the first, second, fourth, etc. steps have indices
        # 0, 1, 3... rather than 1, 2, 4... Similarly, the thousandth
        # step will have index 999 rather than 1000.
        assert is_checkpoint_step(step - 1) is True
    for step in non_checkpoint_steps:
        assert is_checkpoint_step(step - 1) is False
