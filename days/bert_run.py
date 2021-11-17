import torch as t
import numpy as np
from modules import cross_entropy
from einops import rearrange
from days.bert import Bert, my_bert_from_hf_weights
from utils import tpeek, tstat
import transformers
import torchtext
import gin

from torch.optim import Adam

device = "cuda" if t.cuda.is_available() else "cpu"
print("using device", device)


@gin.configurable
def bert_mlm_pretrain(model, tokenizer, dataset, epochs=10, lr=1e-5):
    tokenizer_output = tokenizer(dataset)
    token_ids = t.LongTensor(tokenizer_output["input_ids"])
    model.train()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    num_warmup_steps = 10
    train_context_length = 256
    batch_size = 16
    mask_fraction = 0.15
    trunc_token_ids = token_ids[
        : (token_ids.shape[0] // (batch_size * train_context_length)) * (batch_size * train_context_length)
    ]
    print("have token ids")
    batches = rearrange(trunc_token_ids, "(n b l) -> n b l", b=batch_size, l=train_context_length)
    batches = batches[t.randperm(batches.shape[0])]
    print("batches", batches.shape)
    print_every_n = 2
    print("starting training")
    for epoch in range(epochs):
        for i in range(batches.shape[0]):
            input_ids = batches[i].to(device)
            # mask tokens in sequence, not batches (that's why it's index 1)
            mask_ids = t.FloatTensor(batch_size, train_context_length).to(device).uniform_(0, 1) < mask_fraction
            masked_input_ids = input_ids * ~mask_ids
            masked_input_ids += mask_ids * tokenizer.mask_token_id
            model_output = model(input_ids=masked_input_ids, token_type_ids=t.zeros_like(input_ids).to(device))
            model_output = model_output.logits
            hidden_input_ids = input_ids * mask_ids
            # hidden_input_ids = t.randint(1,1000,input_ids.shape).to(device)
            model_output_flattened = rearrange(model_output, "b s c -> (b s) c")
            hidden_input_ids_flattened = rearrange(hidden_input_ids, "b s -> (b s)")
            loss = cross_entropy(model_output_flattened, hidden_input_ids_flattened, ignore_index=0)
            loss.backward()
            optimizer.step()
            if i % print_every_n == print_every_n - 1:
                print(f"Loss: {loss.cpu().item()}")


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    model = Bert({"hidden_size": 256, "intermediate_size": 1024, "num_layers": 3, "num_heads": 8})
    model = my_bert_from_hf_weights()
    print(model)
    train_data_sentence_iterator = torchtext.datasets.WikiText2(split="valid")
    train_data = "\n".join(train_data_sentence_iterator).replace("<unk>", "[UNK]")

    bert_mlm_pretrain(model, tokenizer, train_data)

# bert mlm training notes: loss you get from predicting random tokens out of 10k is 9.3, 1k is 6.7
