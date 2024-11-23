# simple_stories_train

Project for training small LMs. Designed for training on SimpleStories, an extension of
[TinyStories](https://arxiv.org/abs/2305.07759).


- Training script is based on the efficeint [train_gpt2.py](https://github.com/karpathy/llm.c/blob/master/train_gpt2.py) in [llm.c](https://github.com/karpathy/llm.c) (licensed
  under MIT ((c) 2024 Andrei Karpathy))
- Some model architecture implementations are based on
  [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (licensed under
  MIT ((c) 2022 TransformerLensOrg)).

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

Training a simple model:
`python simple_stories_train/train_llama.py --model d2 --sequence_length 1024 --total_batch_size=4096`

For a final model, we currently (intend to) run:
`torchrun --standalone --nproc_per_node=8 simple_stories_train/train_llama.py --model d24 --sequence_length 1024 --total_batch_size=16448 --compile 1 --tensorcores=1 --dtype=bfloat16 --wandb 1`

You may be asked to enter your wandb API key. You can find it in your [wandb account settings](https://wandb.ai/settings). Alternatively, to avoid entering your API key on program execution, you can set the environment variable `WANDB_API_KEY` to your API key, or put it in a
`.env` file under the root of the repository.
