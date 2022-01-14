import gin
import torch
from torch.utils.data import DataLoader
import torchtext
import transformers

import days.w2d1.bert_sol as bert_sol
import days.w2d1.bert_tests as bert_tests

# this should be True unless you're disabling it for faster iteration while
# debugging stuff
LOAD_PRETRAINED_WEIGHTS = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_bert_without_classification_head_params():
    bert = bert_sol.BertWithClassify(
        vocab_size=28996, hidden_size=768, max_position_embeddings=512, 
        type_vocab_size=2, dropout=0.1, intermediate_size=3072, 
        num_heads=12, num_layers=12, num_classes=2
    ).to(DEVICE)
    if LOAD_PRETRAINED_WEIGHTS:
        pretrained_bert = bert_tests.get_pretrained_bert()
        mapped_params = {bert_sol.mapkey(k): v for k, v in pretrained_bert.state_dict().items()
                        if not k.startswith('classification_head')}
        bert.load_state_dict(mapped_params, strict=False)
    return bert

def get_data(tokenizer, batch_size, max_seq_len):
    data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
    train_dataloader = DataLoader(list(data_train), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(data_test, batch_size=batch_size)
    wrap = lambda data: wrap_loader(data, tokenizer=tokenizer, max_seq_len=max_seq_len) 
    return wrap(train_dataloader), wrap(test_dataloader)

# fetch targets and tokens out of each batch of dataloader
def wrap_loader(dataloader, tokenizer, max_seq_len):
    for targets, inputs in dataloader:
        targets_tensor = torch.tensor(
                [rating == 'pos' for rating in targets],
                dtype=torch.long,
                device=DEVICE,
        )
        tokens = tokenizer(list(inputs), padding='longest', max_length=max_seq_len, truncation=True).input_ids
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
        yield targets_tensor, tokens_tensor

def run_training_loop(model, tokenizer, dataloader, epochs, optimizer, loss_fn, max_seq_len):
    model.train()
    t = 0
    for epoch in range(epochs):
        for targets, inputs in dataloader:
            optimizer.zero_grad()
            _lm_output, classifications = model(inputs)
            loss = loss_fn(classifications, targets)
            loss.backward()
            optimizer.step()

            # t += 1
            # if t % 100 == 0:
            #     print(t, loss)
            #     print("REMINDER: remove this code block before running on rrjobs")
            #     break

def get_test_loss(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    num_items = 0
    t = 0
    with torch.no_grad():
        for targets, inputs in dataloader:
            _lm_output, classifications = model(inputs)
            loss = loss_fn(classifications, targets)

            batch_size = inputs.shape[0]
            num_items += batch_size
            total_loss += batch_size * loss

            # t += 1
            # if t % 100 == 0:
            #     print(t, total_loss / num_items)
    return total_loss / num_items

@gin.configurable
def train(experiment, batch_size, epochs, optimizer, lr, max_seq_len):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    bert = load_bert_without_classification_head_params()
    train_dataloader, test_dataloader = get_data(
            tokenizer=tokenizer,
            batch_size=batch_size, 
            max_seq_len=max_seq_len,
    )
    loss = torch.nn.CrossEntropyLoss()

    if optimizer == "adam":
        optim_fn = torch.optim.Adam
    elif optimizer == "sgd":
        optim_fn = torch.optim.SGD
    optimizer = optim_fn(bert.parameters(), lr)

    run_training_loop(
            model=bert,
            tokenizer=tokenizer,
            dataloader=train_dataloader,
            epochs=epochs,
            optimizer=optimizer,
            loss_fn=loss,
            max_seq_len=max_seq_len
    )

    # test_loss = get_test_loss(
    #         model=bert, 
    #         dataloader=test_dataloader,
    #         loss_fn=loss
    # )

if __name__ == "__main__":
    print("NOTE: running this directly is for debugging. please run this via rrjobs instead\n\n")
    train(batch_size=16, epochs=1, optimizer="adam", lr=1e-5, max_seq_len=512)
    print("done")
