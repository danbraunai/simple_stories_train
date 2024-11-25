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

### Training a model
```
python train_llama.py [PATH/TO/CONFIG.yaml] [--key1 value1 --key2 value2 ...]
```
where
- `PATH/TO/CONFIG.yaml` contains the training config. If no path is provided, a default config will be used.
- `--key1 value1 --key2 value2 ...` override values in the config. Note that if you wish to update a
  nested value, you must use dotted notation (e.g. `--train_dataset_config.name my_dataset`).

If running on CPU, you may need to set `--compile=False`.

To run on multiple GPUs, use
```
torchrun --standalone --nproc_per_node=N train_llama.py ...
```
where `N` is the number of GPUs to use.

### Logging with Weights & Biases
To track training with Weights & Biases, you can set the WANDB_PROJECT and WANDB_API_KEY variables in
`.env`. API keys can be obtained from your [Weights & Biases account settings](https://wandb.ai/settings).
