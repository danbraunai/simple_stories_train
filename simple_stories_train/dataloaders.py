from typing import Any

import numpy as np
import torch
from datasets import Dataset, IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node
from numpy.typing import NDArray
from pydantic import BaseModel
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

"""
The bulk of this file is copied from https://github.com/ApolloResearch/e2e_sae
licensed under MIT, (c) 2024 ApolloResearch.
"""

class DatasetConfig(BaseModel):
    dataset_name: str
    is_tokenized: bool = True
    tokenizer_file_path: str
    streaming: bool = True
    split: str
    n_ctx: int
    seed: int | None = None
    column_name: str = "input_ids"
    ddp_rank: int = 0
    ddp_world_size: int = 1
    """The name of the column in the dataset that contains the data (tokenized or non-tokenized).
    Typically 'input_ids' for datasets stored with e2e_sae/scripts/upload_hf_dataset.py, or "tokens"
    for datasets tokenized in TransformerLens (e.g. NeelNanda/pile-10k)."""


def _keep_single_column(dataset: Dataset, col_name: str) -> Dataset:
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful
    when we want to tokenize and mix together different strings.
    """
    for key in dataset.features:  # pyright: ignore[reportAttributeAccessIssue]
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: Tokenizer,
    max_length: int = 1024,
    column_name: str = "story",
    add_bos_token: bool = False,
    num_proc: int = 10,
    to_lower: bool = True,
) -> Dataset:
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to
    tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of
    shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if
    parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with
    padding, then remove padding at the end.

    NOTE: Adapted from
    https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/utils.py#L267
    to handle IterableDataset.

    TODO: Fix typing of tokenizer

    This tokenization is useful for training language models, as it allows us to efficiently train
    on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding).
    Further, for models with absolute positional encodings, this avoids privileging early tokens
    (eg, news articles often begin with CNN, and models may learn to use early positional
    encodings to predict these)

    Args:
        dataset: The dataset to tokenize, assumed to be a HuggingFace text dataset. Can be a regular
            Dataset or an IterableDataset.
        tokenizer: The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        max_length: The length of the context window of the sequence. Defaults to 1024.
        column_name: The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token: Add BOS token at the beginning of each sequence. Defaults to False as this
            is not done during training.

    Returns:
        Dataset or IterableDataset: Returns the tokenized dataset, as a dataset of tensors, with a
        single column called "input_ids".

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it
    just outputs nothing. I'm not super sure why
    """
    dataset = _keep_single_column(dataset, column_name)
    seq_len = max_length - 1 if add_bos_token else max_length

    def tokenize_function(
        examples: dict[str, list[str]],
    ) -> dict[
        str,
        NDArray[np.signedinteger[Any]],
    ]:
        padding_token_id = 0
        tokenizer.enable_padding(pad_id=padding_token_id)

        text = examples[column_name]
        # Concatenate all the text into a single string, separated by EOS tokens
        full_text = "[EOS]".join(text)

        # Split the text into chunks for parallel tokenization
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]

        # Tokenize the chunks using the Tokenizer library
        if to_lower:
            chunks = [chunk.lower().replace("[eos]", "[EOS]") for chunk in chunks]
        tokens = [
            tokenizer.encode(chunk).ids for chunk in chunks
        ]  # Get token IDs for each chunk
        tokens = np.concatenate(tokens)  # Flatten the list of token IDs

        # Drop padding tokens (if applicable)
        tokens = tokens[tokens != padding_token_id]

        # Calculate number of batches and adjust the tokens accordingly
        num_tokens = len(tokens)
        num_batches = num_tokens // seq_len
        tokens = tokens[: seq_len * num_batches]

        # Reshape tokens into batches
        tokens = tokens.reshape((num_batches, seq_len))

        # Optionally, add BOS token at the beginning of each sequence
        if add_bos_token:
            bos_token_id = tokenizer.token_to_id("[BOS]")
            prefix = np.full((num_batches, 1), bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)

        return {"input_ids": tokens}

    # Apply the tokenization function to the dataset
    if isinstance(dataset, IterableDataset):
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name]
        )
    else:
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name], num_proc=num_proc
        )

    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset



def create_data_loader(
    dataset_config: DatasetConfig, batch_size: int, buffer_size: int = 1000, global_seed: int = 0
) -> tuple[DataLoader[Any], Tokenizer]:
    """Create a DataLoader for the given dataset.

    Args:
        dataset_config: The configuration for the dataset.
        batch_size: The batch size.
        buffer_size: The buffer size for streaming datasets.
        global_seed: Used for shuffling if dataset_config.seed is None.

    Returns:
        A tuple of the DataLoader and the tokenizer.
    """
    dataset = load_dataset(
        dataset_config.dataset_name, streaming=dataset_config.streaming, split=dataset_config.split
    )
    seed = dataset_config.seed if dataset_config.seed is not None else global_seed
    if dataset_config.streaming:
        assert isinstance(dataset, IterableDataset)
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    else:
        dataset = dataset.shuffle(seed=seed)
    dataset = split_dataset_by_node(dataset, dataset_config.ddp_rank, dataset_config.ddp_world_size) # type: ignore
    
    tokenizer = Tokenizer.from_file(dataset_config.tokenizer_file_path)

    torch_dataset: Dataset
    if dataset_config.is_tokenized:
        torch_dataset = dataset.with_format("torch")  # type: ignore
        # Get a sample from the dataset and check if it's tokenized and what the n_ctx is
        # Note that the dataset may be streamed, so we can't just index into it
        sample = next(iter(torch_dataset))[dataset_config.column_name]  # type: ignore
        assert (
            isinstance(sample, torch.Tensor) and sample.ndim == 1
        ), "Expected the dataset to be tokenized."
        assert len(sample) == dataset_config.n_ctx, "n_ctx does not match the tokenized length."

    else:
        torch_dataset = tokenize_and_concatenate(
            dataset,  # type: ignore
            tokenizer,
            max_length=dataset_config.n_ctx,
            add_bos_token=False,
        )

    loader = DataLoader[Any](
        torch_dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=False
    )
    return loader, tokenizer
