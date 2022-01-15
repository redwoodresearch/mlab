"""Utilities for lesswrong data."""

from random import seed
from typing import Iterator, Optional, List, Dict, Generator, Union

import torch as t
from torch.utils.data import DataLoader, TensorDataset
import json
import transformers

import torch.distributed as dist


def pack_texts(texts: List[str], seq_len: int) -> List[List[int]]:
    """
    Takes as input a list of texts.
    Returns a list of id sequences.
    Each returned id sequence has length seq_len.

    The returned id sequences are generated in essentially the following way:
    1. Tokenize texts[i] into tok_seqs[i].
    2. Let big_seq =
            tok_seqs[0] + [END_OF_TEXT_ID]
            + tok_seqs[1] + [END_OF_TEXT_ID]
            + ...
            + tok_seqs[-1]
    3. Divide up big_seq into size seq_len subarrays, and return those subarrays
       as the result. Any tokens at the very end that don't make up a full size
       seq_len subarray are dropped.

    """
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    tokenized = tokenizer(texts)
    eot_token = tokenizer.eos_token_id
    big_seq = []
    for seq in tokenized["input_ids"]:
        big_seq.extend(seq)
        big_seq.append(eot_token)

    # split into seq_len-sized chunks
    seqs = []
    for i in range(0, len(big_seq) - seq_len + 1, seq_len):
        seqs.append(big_seq[i : i + seq_len])
    return seqs


def pretokenize_lesswrong_data(seq_len: int = 1024):
    """
    Pretokenizes the lesswrong data.
    """
    with open("lw_corpus.json", "r") as f:
        lw_corpus: List[Dict] = json.load(f)

    token_seqs = pack_texts(
        [d["text"] for d in lw_corpus],
        seq_len=seq_len,
    )

    token_seqs_tensor = t.tensor(token_seqs, dtype=t.long)

    print(f"Saving token_seqs_tensor of shape {token_seqs_tensor.shape}")
    t.save(token_seqs_tensor, "lw_corpus_tokenized.pt")


class LesswrongDistributedDataLoader:
    def __init__(
        self,
        rank: int,
        world_size: int,
        mini_batch_size: int,
        device: str,
        seq_len: int = 1024,
        random_seed: Optional[int] = 0,
    ):
        self.rank = rank
        self.device = device
        self.mini_batch_size = mini_batch_size
        self.mini_batch_shape = (mini_batch_size, seq_len)
        self.batch_size = world_size * mini_batch_size
        self.batch_shape = (self.batch_size, seq_len)
        self.dataloader: Union[DataLoader, List[t.Tensor]]
        if rank == 0:
            lw_corpus_tokenized = t.load("lw_corpus_tokenized.pt").to(device)
            ds = TensorDataset(lw_corpus_tokenized)
            self.dataloader = DataLoader(
                ds,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
            )
            n_batches_list = [len(self.dataloader)]
        else:
            n_batches_list = [None]

        dist.broadcast_object_list(n_batches_list, src=0)
        self.n_batches = n_batches_list[0]

        if rank != 0:
            self.dataloader = [
                (t.empty(self.batch_shape, dtype=t.long, device=self.device),)
            ] * self.n_batches

    def __iter__(self):
        for (batch,) in self.dataloader:
            dist.broadcast(batch, src=0)

            # minibatch is the self.rank-th chunk of batch
            minibatch = batch[
                self.rank
                * self.mini_batch_size : (self.rank + 1)
                * self.mini_batch_size
            ]
            yield minibatch


if __name__ == "__main__":
    # pretokenize_lesswrong_data()
    # dl = DistributedDataLoader(
    #     rank=0,
    #     world_size=1,
    #     mini_batch_size=16,
    # )
    # print(type(dl.lw_corpus))
    # print(len(dl.lw_corpus))
    # print(dl.lw_corpus[0].keys())

    # for x in dl:
    #     print(x)

    # with open("lw_corpus.json", "r") as f:
    #     lw_corpus: List[Dict] = json.load(f)
    # texts = [d["text"] for d in lw_corpus[:200]]
    # batches = pack_texts(texts, 10)

    # print("\n".join(batches[:100]))
    pass
