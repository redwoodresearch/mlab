from days.w2d4.bert import my_bert_from_hf_weights
import transformers
import gin
import torchtext
import einops
import torch as t


def tokenize_text(text, tokenizer, max_seq_len = 512):
    return tokenizer(text[0], truncation=True, padding='max_length', max_length=max_seq_len)["input_ids"]
    
def preprocess_data(data, tokenizer, batch_size=8, max_seq_len=512):
    labels = [x[0] for x in data]
    text = [x[1] for x in data]
    final_inputs = []
    final_labels = []
    for i in range(len(text)):
        if len(text[i].strip().split(" ")) < 10:
            continue
        else:
            tokenized_text = tokenize_text(text[i], tokenizer, max_seq_len)
            final_inputs.append(tokenized_text)
            final_labels.append(0 if labels[i] == "neg" else 1)
            
    s = len(final_inputs) % batch_size
    final_inputs = einops.rearrange(t.tensor(final_inputs[s:]), "(b k) w -> b k w", k = batch_size)
    final_labels = einops.rearrange(t.tensor(final_labels[s:]), "(b k) -> b k", k = batch_size)
    return final_inputs, final_labels


@gin.configurable
def train(experiment,
          intermediate_size=3072, 
          num_heads=12, 
          head_size=45, 
          context_length=512, 
          num_layers=12,
          dropout=0.1,
          lr=0.01,
          batch_size=8,
          num_epochs=1):
    config = {
        "vocab_size": 28996,
        "intermediate_size": intermediate_size,
        "hidden_size": num_heads * head_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "max_position_embeddings": context_length,
        "dropout": dropout,
        "type_vocab_size": 2,
        "num_classes": 2
    }
    model, _ = my_bert_from_hf_weights(config=config)
    model.to("cuda")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    data_train, data_test = torchtext.datasets.IMDB(root=".data", split=("train", "test"))
    data_train = list(data_train)
    data_test = list(data_test)
    
    train_inputs, train_labels = preprocess_data(data_train, tokenizer, batch_size=batch_size, max_seq_len=context_length)
    
    optimizer = t.optim.Adam(model.parameters(), lr)
    
    for epoch in range(num_epochs):
        for batch_inputs, batch_labels in zip(train_inputs, train_labels):
            
            batch_inputs.to("cuda")
            batch_labels.to("cuda")
            
            out = model(batch_inputs).classification
            
            loss = t.nn.functional.cross_entropy(out, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    