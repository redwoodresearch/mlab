from days.w2d4.attention import *
import torch as t
from torch import einsum
from torch.nn import Module
from einops import rearrange, reduce, repeat
from typing import Callable
from torchtyping import patch_typeguard, TensorType
import numpy as np
import days.w2d1.bert_tests as tests
import torchtext
import gin
import transformers


class ClassBertHead(Module):
    def __init__(self, num_classes, dropout, hidden_size):
        super().__init__()
        self.dropout = t.nn.Dropout(dropout)
        self.classification = t.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids): # batch x seq_len x hidden_size
        first_token = input_ids[:, 0, :]
        return self.classification(self.dropout(first_token))

class ClassificationBert(Module):
    def __init__(self, num_classes, dropout, hidden_size, vocab_size, **kwargs):
        super().__init__()
        self.bert = BaseBert(dropout=dropout, hidden_size=hidden_size, vocab_size=vocab_size, **kwargs)
        self.lmhead = LMBertHead(vocab_size=vocab_size, hidden_size=hidden_size)
        self.classhead = ClassBertHead(num_classes, dropout, hidden_size)

    def forward(self, input_ids):
        bert_output = self.bert(input_ids) 
        class_output = self.classhead(bert_output)
        lm_output = self.lmhead(bert_output)
        return (lm_output, class_output)

def load_pretrained_bert_base(my_bert):
    pretrained_bert = tests.get_pretrained_bert()

    my_keys = my_bert.state_dict().keys()
    their_keys = pretrained_bert.state_dict().keys()

    key_fixed_state_dict = {}
    for my_key, their_key in zip(my_keys, their_keys):
        if my_key.startswith("bert."):
            key_fixed_state_dict[my_key] = pretrained_bert.state_dict()[their_key]

    my_bert.load_state_dict(key_fixed_state_dict, strict=False)


@gin.configurable
def setup_bert(dropout=0.1):
    class_bert = ClassificationBert(num_classes=2,
        vocab_size=28996, hidden_size=768, max_position_embeddings=512, 
        type_vocab_size=2, dropout=dropout, intermediate_size=3072, 
        num_heads=12, num_layers=12
    )
    
    load_pretrained_bert_base(class_bert)
    return class_bert


def extract_data(data, max_seq_length, batch_size, tokenizer):
    shuffled_data = np.array(list(data))
    np.random.shuffle(shuffled_data)

    num_batches = len(shuffled_data) // batch_size

    shuffled_data = shuffled_data[:num_batches * batch_size, :]
    shuffled_data = np.reshape(shuffled_data, (num_batches, batch_size, 2))

    def make_batch(i):
        tokens = tokenizer(shuffled_data[i,:,1].tolist(), padding='longest', max_length=max_seq_length, return_tensors="pt", truncation=True)
        labels = np.array(shuffled_data[i,:,0])
        labels[labels == "neg"] = 0
        labels[labels == "pos"] = 1
        return (tokens, labels.astype(np.int))

    return (make_batch(i) for i in range(num_batches))


def train_batch(model, optimizer, loss_function, batch_tokens, batch_labels):
    optimizer.zero_grad()
    
    tokens_cuda = batch_tokens.input_ids.cuda()

    print("token shape", tokens_cuda.shape)

    outputs = model(tokens_cuda)[1]
        
    targets = t.tensor(batch_labels).cuda()

    print("target shape", targets.shape)

    loss = loss_function(outputs, targets)

    print(loss.item())

    loss.backward()
        
    optimizer.step()


@gin.configurable
def train(experiment, tag, lr, batch_size, max_seq_len):
    if experiment is not None:
        experiment.add_tag(tag)

        experiment.log_parameters({
            "lr": lr,
            "batch_size": batch_size,
            "seed": t.initial_seed(),
        })

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

    data_train, data_test = torchtext.datasets.IMDB(root=".data", split = ("train", "test"))
    train_batches = extract_data(data_train, max_seq_len, batch_size, tokenizer)
    test_batches = extract_data(data_test, max_seq_len, batch_size, tokenizer)

    class_bert = setup_bert()
    class_bert.train()
    class_bert.cuda()

    optimizer = t.optim.Adam(class_bert.parameters(), lr)
    loss_function = t.nn.CrossEntropyLoss()

    epochs = 1
    for i in range(epochs):
        for tokens, labels in train_batches:
            train_batch(class_bert, optimizer, loss_function, tokens, labels)

    class_bert.eval()    
    total_loss = 0
    num_test_batches = 1000 // batch_size
    count = 0

    for tokens, labels in test_batches:
        tokens_cuda = tokens.input_ids.cuda()
        outputs = class_bert(tokens_cuda)[1]
        targets = t.tensor(labels).cuda()
        loss = loss_function(outputs, targets)
        total_loss += loss.item()
        count += 1
        if count >= num_test_batches:
            break
    
    if experiment is not None:
        experiment.log_metric("test_loss", total_loss/count)


if __name__ == "__main__":
    gin.parse_config_file('days/w2d4/bert_finetuning.gin')
    with gin.unlock_config():


        train(experiment=None)