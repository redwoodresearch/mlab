import torch as t
import torch.nn as nn
from torch import einsum
from einops import rearrange, reduce, repeat
import einops
import torchtext
import gin
from bert_gin import my_bert_from_hf_weights

import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
data = list(data_train).copy()

def preprocess(data, tokenizer, max_seq_len, batch_size):
    all_data = []
    labels = []
    for label, text in data[:64]:
        tokenized_text = tokenizer([text], padding='longest', max_length=max_seq_len, truncation=True)["input_ids"][0]
        if len(tokenized_text) < max_seq_len:
            tokenized_text += [0] * (max_seq_len - len(tokenized_text))
        # tokenized_text = tokenized_text[:max_seq_len]
        all_data.append(tokenized_text)
        labels.append(label)
    all_data = t.IntTensor(all_data[:len(all_data) - (len(all_data) % batch_size)]) #t.Tensor
    print("All data", all_data.dtype)
    labels = t.Tensor(list(map(lambda x: 0 if x == "neg" else 1, labels[:len(labels) - (len(labels) % batch_size)])))
    perm = t.randperm(all_data.shape[0])
    all_data = all_data[perm]
    labels = labels[perm]
    all_data = einops.rearrange(all_data, "(k b) m -> k b m", b = batch_size)
    labels = einops.rearrange(labels, "(k b) -> k b", b = batch_size)
    return all_data, labels

classifier = my_bert_from_hf_weights()[0]
training_batches, training_labels = preprocess(data, tokenizer, 512, 2)

@gin.configurable
def train(experiment, epochs=3, lr=1e-5):
    print(lr)
    adam = t.optim.Adam(classifier.parameters(), lr)
    classifier.train()
    classifier.cuda()
    t.cuda.empty_cache()
    num_batches = training_batches.shape[0]
    batch_size = training_batches.shape[1]
    for epoch in range(epochs):
        print("epoch", epoch)
        for batch_num in range(num_batches):
            adam.zero_grad()
            b = training_batches[batch_num].cuda()
            l = training_labels[batch_num].cuda()
            out = classifier(b).classification #classifier(b)[1]
            out_loss = nn.functional.cross_entropy(out, l.long())
            out_loss.backward()
            adam.step()
            if batch_num % 20 == 0:
                print("batch", batch_num, "loss", out_loss.item())