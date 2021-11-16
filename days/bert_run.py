import torch as t
import numpy as np
from modules import cross_entropy_loss
from torchtyping import TensorType
from einops import rearrange
from days.bert import Bert
from utils import tpeek, tstat
import transformers
import torchtext
import gin

from torch.optim import Adam


@gin.configurable
def train_from_scratch(model, tokenizer, dataset, epochs=2, lr=0.001):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)

    train_context_length = 256
    batch_size = 16
    mask_fraction = 0.15
    tokenizer_output = tokenizer(dataset)

    token_ids = t.LongTensor(tokenizer_output["input_ids"])
    token_ids = token_ids[
        : (token_ids.shape[0] // (batch_size * train_context_length)) * (batch_size * train_context_length)
    ]
    print("have token ids")
    batches = rearrange(token_ids, "(n b l) -> n b l", b=batch_size, l=train_context_length)
    print("batches", batches.shape)
    for epoch in range(epochs):
        for i in range(batches.shape[0]):
            input_ids = batches[i]
            print("input ids shape", input_ids.shape)
            # mask tokens in sequence, not batches (that's why it's index 1)
            mask_ids = t.FloatTensor(batch_size, train_context_length).uniform_(0, 1) < mask_fraction
            masked_input_ids = input_ids * ~mask_ids
            masked_input_ids += mask_ids * tokenizer.mask_token_id
            print("masked input ids", masked_input_ids[0])
            model_output = model(token_ids=masked_input_ids, token_type_ids=t.zeros_like(input_ids))
            tpeek("model output", model_output)
            hidden_input_ids = input_ids * mask_ids

            loss = cross_entropy_loss(model_output, hidden_input_ids, dim=-1, ignore_id=0)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    model = Bert(
        {
            "hidden_size": 256,
            "num_layers": 6,
            "num_heads": 8,
        }
    )
    train_data_sentence_iterator = torchtext.datasets.WikiText2(split="valid")
    train_data = "\n".join(train_data_sentence_iterator).replace("<unk>", "[UNK]")

    train_from_scratch(model, tokenizer, train_data)
