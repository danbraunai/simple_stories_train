import datasets
import numpy as np
import tiktoken
import torch
from jaxtyping import Int


class StreamingDataLoader:
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        sequence_length: int,
        num_processes=1,
        process_rank=0,
        fetch_size=4,
        split: str = "train",
    ):
        self.dataset: datasets.IterableDataset = datasets.load_dataset(  # type:ignore
            dataset_name, split=split, streaming=True, trust_remote_code=True
        )
        self.batch_size = batch_size
        self.encoding = tiktoken.get_encoding("gpt2")
        self.sequence_length = sequence_length
        self.num_processes = num_processes
        self.process_rank = process_rank
        self.fetch_size = fetch_size
        self.buffer = []
        self._prefill_buffer()
        self.seed = process_rank

    def _prefill_buffer(self):
        """
        Gets one sample from the dataset loader and skips
        """
        try:
            batch = self.encode(list(self.dataset.take(self.fetch_size))[0]["text"])
            self.buffer.extend(batch)
            self.dataset.skip(self.fetch_size * (self.num_processes - 1))
        except StopIteration:
            self.dataset.shuffle(seed=self.seed)
            self.seed += self.num_processes

    def next_batch(self) -> tuple[Int[torch.Tensor, "batch pos"], Int[torch.Tensor, "batch pos"]]:
        B = self.batch_size
        T = self.sequence_length

        while len(self.buffer) < B * T + 1:
            self._prefill_buffer()

        buf = np.array(self.buffer[: B * T + 1])
        self.buffer = self.buffer[B * T :]

        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets

        return x, y

    def encode(self, s: str):
        return self.encoding.encode(s, allowed_special={"<|endoftext|>"})

    def reset(self):
        self.iterator = iter(self.dataset)
        self.buffer = []
        self._prefill_buffer()
