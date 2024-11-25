from simple_stories_train.utils import convert_dotted_args_to_nested_dict, is_checkpoint_step


def test_is_checkpoint_step() -> None:
    checkpoint_steps = [1, 2, 4, 8, 512, 1000, 2000, 10000]
    non_checkpoint_steps = [3, 5, 1024, 2048]
    for step in checkpoint_steps:
        assert is_checkpoint_step(step) is True
    for step in non_checkpoint_steps:
        assert is_checkpoint_step(step) is False


def test_convert_dotted_args_to_nested_dict() -> None:
    # Test basic nested structure
    args = {"a.b.c": 1, "a.b.d": 2, "a.e": 3, "f": 4}
    expected = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": 4}
    assert convert_dotted_args_to_nested_dict(args) == expected

    # Test with empty dict
    assert convert_dotted_args_to_nested_dict({}) == {}

    # Test multiple values at same level
    args = {
        "model.name": "gpt2",
        "model.layers": 12,
        "training.enabled": True,
        "simple_key": "value",
    }
    expected = {
        "model": {"name": "gpt2", "layers": 12},
        "training": {"enabled": True},
        "simple_key": "value",
    }
    assert convert_dotted_args_to_nested_dict(args) == expected
