"""Finetuning bert on sentiment classification."""

import comet_ml

# This block forces comet_ml to be the first import
if True:
    pass

from typing import List, Tuple

import gin
import torch as t
import torch.nn.functional as F
import torchtext
import transformers
from days.w2d4.bert import DEFAULT_CONFIG, BertWithClassify
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_imdb_collate_fn(
    max_seq_length: int,
    tokenizer: transformers.AutoTokenizer,
    device: str,
):
    def fn(raw_xs: List[Tuple[str, str]]) -> Tuple[t.Tensor, t.Tensor]:
        labels: Tuple[str, ...]
        texts: Tuple[str, ...]
        labels, texts = zip(*raw_xs)

        xs = t.tensor(
            tokenizer(
                list(texts),
                padding="longest",
                max_length=max_seq_length,
                truncation=True,
            )["input_ids"],
            dtype=t.long,
            device=device,
        )

        ys = t.tensor([int(l == "pos") for l in labels], dtype=t.long, device=device)

        return xs, ys

    return fn


def get_accuracy(
    bert: BertWithClassify,
    dl: DataLoader,
) -> float:
    num_correct: int = 0
    num_total: int = 0

    pbar = tqdm(dl, disable=True)
    for x, y in pbar:
        with t.no_grad():
            _, out = bert.forward(x)
            preds = t.argmax(out, dim=-1)

        print(y)

        num_correct += (preds == y).sum()
        num_total += len(y)
        pbar.set_description(f"acc={num_correct / num_total:.2}")

    return num_correct / num_total


@gin.configurable
def train(
    experiment: comet_ml.Experiment,
    train_batch_size: int,
    test_batch_size: int,
    num_epochs: int,
    lr: float,
) -> BertWithClassify:
    data_train_gen, data_test_gen = torchtext.datasets.IMDB(
        root=".data", split=("train", "test")
    )
    data_train = list(data_train_gen)
    data_test = list(data_test_gen)
    indices = t.randint(0, len(data_train), size=(100,))
    data_train = [data_train[i] for i in indices]
    data_test = [data_test[i] for i in indices]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    collate_fn = get_imdb_collate_fn(512, tokenizer, device="cuda")

    # dl_train_small = DataLoader(
    #     random.sample(data_train, k=train_batch_size),
    #     batch_size=train_batch_size,
    #     collate_fn=collate_fn,
    #     shuffle=True,
    # )

    # dl_test_small = DataLoader(
    #     random.sample(data_test, k=test_batch_size),
    #     batch_size=test_batch_size,
    #     collate_fn=collate_fn,
    #     shuffle=True,
    # )

    dl_train = DataLoader(
        data_train,
        batch_size=train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    dl_test = DataLoader(
        data_test,
        batch_size=test_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    bert = BertWithClassify.pretrained()
    bert.cuda()

    bert.train()
    optimizer = optim.Adam(bert.parameters(), lr=lr)  # broken?
    for epoch in range(num_epochs):

        for x, y in tqdm(dl_train):
            optimizer.zero_grad()
            _, out = bert(x)
            loss = F.cross_entropy(input=out, target=y)
            loss.backward()
            optimizer.step()

        bert.eval()
        test_acc = get_accuracy(bert, dl_test)
        train_acc = get_accuracy(bert, dl_train)
        bert.train()

        experiment.log_metric(name="test_acc", value=test_acc, step=epoch)
        experiment.log_metric(name="train_acc", value=train_acc, step=epoch)

    return bert


if __name__ == "__main__":
    train()
