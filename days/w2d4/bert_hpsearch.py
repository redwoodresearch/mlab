#!/usr/bin/env python3
import gin
import random
import torch as t
from tqdm import tqdm
import torchtext
import transformers

from days.w2d1 import bert_sol

device = 'cuda'

@gin.configurable
def configurable_bert(
        vocab_size,
        hidden_size,
        max_position_embeddings,
        dropout,
        intermediate_size,
        num_heads,
        num_layers,
    ):
    return bert_sol.BertWithClassify(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=2,
        dropout=dropout,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=2,
    )

def get_imdb_batches(which, batch_size, max_seq_len, tokenizer):
    (data,) = torchtext.datasets.IMDB(root='.data', split=(which,))
    data = list(data)
    data.sort(key=lambda d: len(d[1]))
    num_batches = (len(data) + batch_size - 1) // batch_size
    batched_data = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch = data[batch_start:batch_end]
        for s, r in batch:
            assert r
        sentiments = t.tensor([1 if s == 'pos' else 0 for s, r in batch])
        reviews = [r for s, r in batch]
        tokenization = tokenizer(reviews, padding='longest', max_length=max_seq_len, truncation=True)
        review_tokens = t.tensor(tokenization.input_ids)
        batched_data.append((review_tokens, sentiments))
    random.shuffle(batched_data)
    return batched_data

@gin.configurable
def train(model, tokenizer, max_seq_len, lr, batch_size, num_epochs):
    model.train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    data_batches = get_imdb_batches(
        which='train',
        batch_size=batch_size,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    
    for _ in range(num_epochs):
        for i, (input, target) in enumerate(tqdm(data_batches)):
            optimizer.zero_grad()
            logits, class_logits = model(input.to(device))
            loss = t.nn.functional.cross_entropy(class_logits, target.to(device))
            loss.backward()
            optimizer.step()

def main(experiment):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    max_seq_len = 512
    model = configurable_bert(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_seq_len,
    ).to(device)
    train(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

if __name__ == '__main__':
    with gin.unlock_config():
        gin.parse_config_file('bert_train.gin')
    main()