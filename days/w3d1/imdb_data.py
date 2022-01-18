"""Utilities for IMDB data."""

import dataclasses
from typing import List, Optional, Tuple

import torch as t
import torch.distributed as dist
import torchtext
import transformers
from torch.utils.data import DataLoader, TensorDataset


class DistributedIMDBData:
    def __init__(
        self,
        batch_size: int,
        device: str,
    ):
        rank = dist.get_rank()

        self.train_dl: Optional[DataLoader] = None
        if rank == 0:
            training_data_xs = t.load("imdb_train_xs_1024.pt").to(device)
            training_data_ys = t.load("imdb_train_ys.pt").to(device)
            train_ds = TensorDataset(training_data_xs, training_data_ys)
            self.train_dl = DataLoader(
                train_ds,
                batch_size=batch_size,
                drop_last=True,
                shuffle=True,
            )

            # TODO: Load test data as well

        if rank == 0:
            _num_batches_list = [len(training_data_ys) // batch_size]
        else:
            _num_batches_list = [None]
        dist.broadcast_object_list(_num_batches_list, src=0)
        self.num_train_batches: int = _num_batches_list[0]

    def __iter__(self):
        if dist.get_rank() == 0:
            for x in self.train_dl:
                yield x
        else:
            for _ in range(self.num_train_batches):
                yield None, None


def tokenize_and_pad(
    texts: List[str],
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_len: int,
    pad_token_id: int,
) -> t.Tensor:
    tokenized = tokenizer(
        texts,
        max_length=seq_len,
        truncation=True,
    )["input_ids"]

    data = t.full(size=(len(texts), seq_len), fill_value=pad_token_id).long()

    for i, sentence in enumerate(tokenized):
        assert 0 <= len(sentence) <= seq_len
        tensorized_sentence = t.tensor(sentence).long()

        if len(sentence) == seq_len:
            data[i] = tensorized_sentence
        else:
            data[i][-len(sentence) :] = tensorized_sentence
            # data[i][-len(sentence) - 1] = tokenizer.eos_token_id

    return data


@dataclasses.dataclass
class IMDBSplit:
    xs: t.Tensor
    ys: t.Tensor


def tensorize_data(
    data: List[Tuple[str, str]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_len: int,
    pad_token_id: int,
) -> IMDBSplit:
    labels, texts = zip(*data)
    return IMDBSplit(
        xs=tokenize_and_pad(
            list(texts),
            tokenizer=tokenizer,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        ),
        ys=t.tensor([(l == "pos") for l in labels], dtype=t.long),
    )


def imdb_data_for_gptj(seq_len: int) -> Tuple[IMDBSplit, IMDBSplit]:
    # This is slow, beware! ~20 seconds
    print("Begin getting IMDB...", end="")
    data_train, data_test = torchtext.datasets.IMDB(
        root=".data", split=("train", "test")
    )
    data_train = list(data_train)
    data_test = list(data_test)
    print("done")

    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    print("Tensorizing train split...", end="")
    train_split = tensorize_data(
        data_train, tokenizer=tokenizer, seq_len=seq_len, pad_token_id=0
    )
    print("done")

    print("Tensorizing test split...", end="")
    test_split = tensorize_data(
        data_test, tokenizer=tokenizer, seq_len=seq_len, pad_token_id=0
    )
    print("done")

    return train_split, test_split


if __name__ == "__main__":
    imdb_train, imdb_test = imdb_data_for_gptj(seq_len=512)

    t.save(imdb_train.xs, "imdb_train_xs_128.pt")
    t.save(imdb_train.ys, "imdb_train_ys.pt")
    t.save(imdb_test.xs, "imdb_test_xs_128.pt")
    t.save(imdb_test.ys, "imdb_test_ys.pt")
