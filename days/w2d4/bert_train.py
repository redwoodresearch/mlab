from comet_ml import Experiment
from bert_sol import BertWithClassify
import transformers
import torch as t
import gin
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader


@gin.configurable
def train(seed, epochs, lr, **kwargs):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    DEVICE = "cuda"
    experiment = Experiment(
        project_name="dane-beth-w2d4", api_key="CSKZCzO65mpEjuqILZVslr5BF"
    )

    t.manual_seed(seed)
    model = BertWithClassify(
        vocab_size=kwargs["vocab_size"],
        intermediate_size=kwargs["intermediate_size"],
        hidden_size=kwargs["hidden_size"],
        num_layers=kwargs["num_layers"],
        num_heads=kwargs["num_heads"],
        max_position_embeddings=kwargs["max_position_embeddings"],
        dropout=kwargs["dropout"],
        type_vocab_size=kwargs["type_vocab_size"],
        num_classes=kwargs["num_classes"],
    ).to(DEVICE)
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # get IMDB data from torchtext
    training_data, test_data = IMDB(root=".data", split=("train", "test"))

    # TODO: shuffle these later
    train_dataloader = DataLoader(training_data, batch_size=8)
    test_dataloader = DataLoader(test_data, batch_size=64)

    for _ in range(epochs):
        for y, x in list(train_dataloader)[:5]:
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


if __name__ == "__main__":
    with gin.unlock_config():
        gin.parse_config_file(config_file="bert_train.gin")
        train()
