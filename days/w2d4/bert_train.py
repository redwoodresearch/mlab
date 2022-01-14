from comet_ml import Experiment
import bert_sol
import transformers
import torch as t
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
DEVICE = "cuda"
experiment = Experiment(
    project_name="dane-beth-w2d4", api_key="CSKZCzO65mpEjuqILZVslr5BF"
)


def train(seed, epochs=1, **kwargs):
    t.manual_seed(seed)
    model = bert_sol.BertWithClassify(**kwargs).to(DEVICE)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    # get IMDB data from torchtext
    training_data, test_data = IMDB(root=".data", split=("train", "test"))

    # TODO: shuffle these later
    train_dataloader = DataLoader(training_data, batch_size=8)
    test_dataloader = DataLoader(test_data, batch_size=64)

    for epoch in range(epochs):
        for y, x in list(train_dataloader):
            labels = t.tensor([1 if label == "pos" else 0 for label in y]).to(DEVICE)
            tokens = tokenizer(
                list(x),
                padding="longest",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = t.tensor(tokens["input_ids"]).to(DEVICE)
            logits, classification_logits = model(input_ids)
            loss = t.nn.functional.cross_entropy(classification_logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            experiment.log_metric("loss", loss.item())


config = {
    "vocab_size": 28996,
    "intermediate_size": 3072,
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "max_position_embeddings": 512,
    "dropout": 0.1,
    "type_vocab_size": 2,
    "num_classes": 2,
}


if __name__ == "__main__":
    train(seed=42, **config)
