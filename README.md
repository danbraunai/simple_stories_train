# simple_stories_train

Project for training small LMs. Designed for training on SimpleStories, an extension of [TinyStories](https://arxiv.org/abs/2305.07759).

## Installation

From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements and pre-commit hooks
make install  # To just install the package (runs `pip install -e .`)
```

## Development

Suggested extensions and settings for VSCode are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

There are various `make` commands that may be helpful

```bash
make check  # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```

## Usage
First, download the TinyShakespeare dataset:
```bash
python simple_stories_train/tinyshakespeare.py
```

Then, train llama on it:
`python simple_stories_train/train_llama.py --model d12 --input_bin simple_stories_train/tinyshakespeare/tiny_shakespeare_val.bin`

You may be asked to enter your wandb API key. You can find it in your [wandb account settings](https://wandb.ai/settings). Alternatively, to avoid entering your API key on program execution, you can set the environment variable `WANDB_API_KEY` to your API key.
