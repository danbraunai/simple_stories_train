"""
TODO: Use the finalized tokenizer.
This file is copied almost 1:1 from Nix Goldowsky-Dill's adaption of the tokenizer in https://github.com/juand-r/tiny_tokenizer.
"""

from itertools import chain
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Digits, Punctuation, Whitespace
from tokenizers.trainers import WordPieceTrainer

OUT_DIR = Path("tokenizers")


def clean_dataset():
    dataset = load_dataset("lennart-finke/SimpleStories")
    trans = str.maketrans(
        {"\u201d": '"', "\u201c": '"', "\u2019": "'", "\u2018": "'", "\u2014": "-", "\u2026": "..."}
    )
    cleaned = [
        s.translate(trans).encode("ascii", "ignore").decode("ascii").lower()
        for s in dataset["train"]["story"] # pyright: ignore
    ]

    n_train = int(len(cleaned) * 0.9) 
    train, validation = cleaned[:n_train], cleaned[n_train:]

    train_ds = Dataset.from_dict(dict(story=train))
    validation_ds = Dataset.from_dict(dict(story=validation))

    return DatasetDict({"train": train_ds, "validation": validation_ds})


def train_tokenizer(vocab_size=4096):
    data = clean_dataset()["train"]["story"]

    # Split almost all types into individual tokens
    pre_tokenizer = pre_tokenizers.Sequence(
        [Punctuation(), Whitespace(), Digits(individual_digits=True)]
    )

    # The tokenizer itself, being a WordPiece tokenizer, which is generally smaller than a byte pair encoding
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]")) # pyright: ignore
    tokenizer.pre_tokenizer = pre_tokenizer # pyright: ignore

    print("training tokenizer...")
    # Train the tokenizer
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[EOS]"])
    tokenizer.train_from_iterator(data, trainer=trainer, length=len(data))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(f"{OUT_DIR}/stories-{vocab_size}.json")


def load_tokenizer(vocab_size=3072) -> Tokenizer:
    return Tokenizer.from_file(f"{OUT_DIR}/stories-{vocab_size}.json")


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


def analysis(vocab_size=3072) -> None:
    tokenizer = load_tokenizer(vocab_size)
    stories = clean_dataset()["validation"]["story"]
    tokenized_stories = [tokenizer.encode(story).tokens for story in stories]

    all_tokens = list(chain.from_iterable(tokenized_stories))
    partial_word_toks = len([token for token in all_tokens if token.startswith("##")])

    print(f"Tokens per story: {len(all_tokens) / len(stories):.2f}")
    print(f"Proportion of partial word tokens: {partial_word_toks / len(all_tokens):.2%}")

    for story_idx, story_tokens in enumerate(tokenized_stories[:25]):
        print(f"\nStory {story_idx}")
        print_split_words(story_tokens)