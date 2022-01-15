import torch as t
import numpy as np
from modules import cross_entropy
from einops import rearrange
from days.bert import Bert, my_bert_from_hf_weights
from days.utils import tpeek
import transformers
import torchtext
import gin
from datetime import datetime
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
        : (token_ids.shape[0] // (batch_size * train_context_length))
        * (batch_size * train_context_length)
    ]
    print("have token ids")
    batches = rearrange(
        trunc_token_ids, "(n b l) -> n b l", b=batch_size, l=train_context_length
    )
    batches = batches[t.randperm(batches.shape[0])]
    print("batches", batches.shape)
    print_every_n = 10
    save_every_n = 200
    print("starting training")
    for epoch in range(epochs):
        for i in range(batches.shape[0]):
            input_ids = batches[i].to(device)
            # mask tokens in sequence, not batches (that's why it's index 1)
            mask_ids = (
                t.FloatTensor(batch_size, train_context_length)
                .to(device)
                .uniform_(0, 1)
                < mask_fraction
            )
            masked_input_ids = input_ids * ~mask_ids
            masked_input_ids += mask_ids * tokenizer.mask_token_id
            model_output = model(
                input_ids=masked_input_ids,
                token_type_ids=t.zeros_like(input_ids).to(device),
            ).logits
            if t.any(t.isnan(model_output)):
                print("NAN output!!!!!")
            # model_output = model_output.logits
            hidden_input_ids = input_ids * mask_ids
            # hidden_input_ids = t.randint(1,1000,input_ids.shape).to(device)
            model_output_flattened = rearrange(model_output, "b s c -> (b s) c")
            hidden_input_ids_flattened = rearrange(hidden_input_ids, "b s -> (b s)")
            loss = cross_entropy(
                model_output_flattened, hidden_input_ids_flattened, ignore_index=0
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % print_every_n == print_every_n - 1:
                print(f"Loss: {loss.cpu().item()}")
            if (i + epoch * batches.shape[0]) % save_every_n == save_every_n - 1:

                now = datetime.now()
                date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = f".my_bert_{date_time}_epoch{epoch}"
                t.save(model, file_name)
                print(f"finished epoch {epoch} saving to {file_name}")


def ids_to_strings(tokenizer, ids):
    token_strings = tokenizer.convert_ids_to_tokens(ids)
    token_strings = [
        tokenizer.convert_tokens_to_string([string]) for string in token_strings
    ]
    return token_strings


def infer_bert(model, tokenizer, text):
    input_ids = t.LongTensor(tokenizer(text).input_ids).unsqueeze(0)
    logits = model(input_ids=input_ids).logits.squeeze(0)
    return logits


def infer_show_bert(model, tokenizer, text):
    tokens = tokenizer(text).input_ids
    mask_idx = tokens.index(tokenizer.mask_token_id)
    print(mask_idx)
    logits = infer_bert(model, tokenizer, text)
    top10 = t.topk(logits[mask_idx], 10).indices
    topk_words = [ids_to_strings(tokenizer, [tok])[0] for tok in top10]
    print(topk_words)


def eval_bert_mlm(model, dataset):
    fun_texts = [
        "my name is Amy. Also, my name is [MASK].",
        "Hello, my name is [MASK].",
        "[MASK], my name is Amy.",
    ]

    for fun_text in fun_texts:
        infer_show_bert(model, tokenizer, fun_text)


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    model = Bert(
        {"hidden_size": 256, "intermediate_size": 1024, "num_layers": 3, "num_heads": 8}
    )

    model = my_bert_from_hf_weights()
    print(model)
    train_data_sentence_iterator = torchtext.datasets.WikiText2(split="valid")
    train_data = "\n".join(train_data_sentence_iterator).replace("<unk>", "[UNK]")

    val_data_sentence_iterator = torchtext.datasets.WikiText2(split="valid")
    val_data = "\n".join(val_data_sentence_iterator).replace("<unk>", "[UNK]")

    bert_mlm_pretrain(model, tokenizer, train_data)

    eval_bert_mlm(model, val_data)
# bert mlm training notes: loss you get from predicting random tokens out of 10k is 9.3, 1k is 6.7
