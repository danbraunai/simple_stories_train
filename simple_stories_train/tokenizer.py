"""
This file is inspired from Nix Goldowsky-Dill's adaption of the tokenizer in https://github.com/juand-r/tiny_tokenizer.
"""

from itertools import chain
from pathlib import Path

from dataloaders import DatasetConfig, create_data_loader
from datasets import Dataset, DatasetDict, load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, Replace
from tokenizers.normalizers import Sequence as NormSequence
from tokenizers.pre_tokenizers import Digits, Punctuation, Sequence, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

OUT_DIR = Path("tokenizer")

# Define common affixes for special handling based on morphological analysis of the dataset
COMMON_PREFIXES = ["un", "re"]
COMMON_SUFFIXES = ["ed", "ing", "ly", "er", "ness"]


def clean_dataset(dataset="lennart-finke/SimpleStories") -> DatasetDict:
    """
    Load and clean the dataset, implementing lowercase strategy.
    """
    dataset = load_dataset(dataset, trust_remote_code=False)
    trans = str.maketrans(
        {"\u201d": '"', "\u201c": '"', "\u2019": "'", "\u2018": "'", "\u2014": "-", "\u2026": "..."}
    )

    cleaned = [
        s.translate(trans).encode("ascii", "ignore").decode("ascii").lower()
        for s in dataset["train"]["story"]  # pyright: ignore
    ]

    # Split into train and validation sets
    n_train = int(len(cleaned) * 0.9)
    train, validation = cleaned[:n_train], cleaned[n_train:]

    train_ds = Dataset.from_dict(dict(story=train))
    validation_ds = Dataset.from_dict(dict(story=validation))

    return DatasetDict({"train": train_ds, "validation": validation_ds})


def create_tokenizer(vocab_size=4096) -> Tokenizer:
    """
    Create a tokenizer with integrated affix handling using Split pre-tokenizers.

    Args:
        vocab_size: The target vocabulary size for the tokenizer

    Returns:
        A configured Tokenizer object ready for training
    """
    print(f"Creating tokenizer with target vocabulary size: {vocab_size}")

    # Initialize WordPiece tokenizer with vocabulary size hint
    tokenizer = Tokenizer(
        WordPiece(
            unk_token="[UNK]",
            vocab_size=vocab_size,  # type: ignore
        )
    )

    # Set normalizers (lowercase everything)
    tokenizer.normalizer = NormSequence(
        [
            Lowercase(),
            Replace("``", '"'),
            Replace("''", '"'),
        ]  # type: ignore
    )

    # Set up the pre-tokenizer sequence
    tokenizer.pre_tokenizer = Sequence(
        [Whitespace(), Punctuation(), Digits(individual_digits=True)]
    )  # type: ignore

    # Add post-processor for special tokens
    tokenizer.post_processor = TemplateProcessing(single="$A [EOS]", special_tokens=[("[EOS]", 1)])  # type: ignore

    tokenizer.decoder = WordPieceDecoder(prefix="##")  # type: ignore

    return tokenizer


def train_tokenizer(vocab_size=4096, dataset="lennart-finke/SimpleStories") -> None:
    """
    Train the tokenizer with the specified vocabulary size and dataset.

    Args:
        vocab_size: The target vocabulary size
        dataset: The dataset to train on
    """
    data = clean_dataset(dataset=dataset)["train"]["story"]

    tokenizer = create_tokenizer(vocab_size)

    special_tokens = ["[UNK]", "[EOS]"]
    affixes = COMMON_PREFIXES + COMMON_SUFFIXES

    print("Training tokenizer...")
    # Train the tokenizer
    trainer = WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=special_tokens, initial_alphabet=affixes
    )

    tokenizer.train_from_iterator(data, trainer=trainer, length=len(data))

    # Save the tokenizer
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = f"{OUT_DIR}/simplestories-{vocab_size}.json"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")


def test_tokenizer(filepath: str, dataset: str = "lennart-finke/SimpleStories") -> None:
    """
    Test the trained tokenizer on sample data.
    """
    dataset_name = dataset
    split = "train"

    context_width = 512
    dataset_config = DatasetConfig(
        name=dataset_name,
        is_tokenized=False,
        tokenizer_file_path=filepath,
        streaming=True,
        split=split,
        n_ctx=context_width,
        seed=42,
        column_name="story",
    )

    batch_size = 1
    buffer_size = 1000
    global_seed = 0

    loader, tokenizer = create_data_loader(
        dataset_config, batch_size, buffer_size, global_seed, ddp_rank=0, ddp_world_size=1
    )
    batch = next(iter(loader))
    words = tokenizer.decode_batch(batch["input_ids"].tolist(), skip_special_tokens=False)
    print("Sample tokenization:")
    print(words)


def load_tokenizer(vocab_size=4096) -> Tokenizer:
    return Tokenizer.from_file(f"{OUT_DIR}/simplestories-{vocab_size}.json")


def print_split_words(story_tokens: list[str]) -> None:
    for i, token in enumerate(story_tokens):
        if token.startswith("##") and (i == 0 or not story_tokens[i - 1].startswith("##")):
            word_start = story_tokens[i - 1] if i > 0 else ""
            word_parts = [token]

            for next_token in story_tokens[i + 1 :]:
                if not next_token.startswith("##"):
                    break
                word_parts.append(next_token)

            print(f"{word_start} {' '.join(word_parts)}")


def analysis(vocab_size=4096) -> None:
    tokenizer = load_tokenizer(vocab_size)
    stories = clean_dataset(dataset="lennart-finke/SimpleStories")["validation"]["story"]
    tokenized_stories = [tokenizer.encode(story).tokens for story in stories]

    all_tokens = list(chain.from_iterable(tokenized_stories))
    partial_word_toks = len([token for token in all_tokens if token.startswith("##")])

    print(f"Tokens per story: {len(all_tokens) / len(stories):.2f}")
    print(f"Proportion of partial word tokens: {partial_word_toks / len(all_tokens):.2%}")

    for story_idx, story_tokens in enumerate(tokenized_stories[:25]):
        print(f"\nStory {story_idx}")
        print_split_words(story_tokens)


if __name__ == "__main__":
    test_tokenizer("tokenizer/simplestories-4096.json")
