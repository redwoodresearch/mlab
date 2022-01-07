import os
import json

if "PARAMS" in os.environ:
    print(os.environ["PARAMS"])
    if "RR_JOBS" in json.loads(os.environ["PARAMS"]):
        os.system("pip install -r ../../requirements.txt")
from comet_ml import Experiment

from typing import Callable, Dict, Iterable, Tuple, Any

import einops
import gin
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import w1d4_tests
from optim import MLABOptim


EXPERIMENT = Experiment(
    api_key="qjxcybqq2HsGHbEwATgNiqWgE",
    project_name="mlab_w1d4_v3",
    workspace="ttwang",
    auto_metric_logging=False,
)


@gin.configurable
class MyNet(nn.Module):
    def __init__(self, P: int, H: int, K: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=P, out_features=H),
            nn.ReLU(),
            nn.Linear(in_features=H, out_features=H),
            nn.ReLU(),
            nn.Linear(in_features=H, out_features=K),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def device(self):
        device = next(self.parameters()).device
        return device


def get_image_from_fn(
    model,
    H: int = 200,
    W: int = 200,
    xstart=-0.5,
    xend=0.5,
    ystart=-0.5,
    yend=0.5,
) -> t.Tensor:
    hs = t.linspace(start=xstart, end=xend, steps=H)
    ws = t.linspace(start=ystart, end=yend, steps=W)

    expanded_hs = einops.repeat(hs, "h -> h w", w=W)
    expanded_ws = einops.repeat(ws, "w -> h w", h=H)

    coords = t.stack((expanded_hs, expanded_ws), dim=-1).to(model.device)

    return t.clip(model(coords) + 0.5, 0, 1), coords


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    get_opt: Callable[[Iterable[t.Tensor]], MLABOptim],
) -> nn.Module:

    optimizer = get_opt(model.parameters())
    loss_fn = F.l1_loss

    for input, target in dataloader:
        optimizer.zero_grad()

        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    return model


def evaluate(model: nn.Module, dataloader: DataLoader) -> t.Tensor:
    loss_fn = F.l1_loss

    total_loss = t.tensor(0.0).to(model.device)
    num_datapoints: int = 0
    for input, target in dataloader:
        output = model(input)
        loss = loss_fn(output, target)

        cur_datapoints = len(target)

        total_loss += loss * cur_datapoints
        num_datapoints += cur_datapoints

    return total_loss / num_datapoints


@gin.configurable
def train(
    n_train: int,
    n_epochs: int,
    batch_size: int,
    optimizer: MLABOptim,
    device: str,
):

    model = MyNet().to(device)
    data_train, data_test = w1d4_tests.load_image(
        "gorilla.jpg",
        device=device,
        batch_size=batch_size,
        n_train=n_train,
    )

    for epoch in range(n_epochs):
        model = train_epoch(
            model,
            data_train,
            lambda params: optimizer(params),
        )

        train_loss = float(evaluate(model, data_train))
        test_loss = float(evaluate(model, data_test))

        #if epoch % 50 == 0:
        print(f"{epoch=}; train={train_loss}; test={test_loss}")
            # print(model.device)

        EXPERIMENT.log_metric(name="train_loss", value=train_loss, step=epoch)
        EXPERIMENT.log_metric(name="test_loss", value=test_loss, step=epoch)

    with t.no_grad():
        img = get_image_from_fn(model, H=64, W=64)[0].cpu().numpy()
        EXPERIMENT.log_image(img)

        # plt.plot(train_losses, label="train")
        # plt.plot(test_losses, label="test")
        # plt.xlabel("epoch")
        # plt.ylabel("L1_loss")
        # plt.legend()
        # plt.show()


def flatten_gin_config(
    d: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    ret_dict = {}
    for (_, sub_name), sub_dict in d.items():
        small_sub_name = sub_name.split(".")[-1]
        ret_dict.update({f"{small_sub_name}.{k}": v for k, v in sub_dict.items()})
    return ret_dict


if __name__ == "__main__":
    with gin.unlock_config():
        params = json.loads(os.environ.get("PARAMS", "{}"))
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, 'config.gin' )

        gin.parse_config_files_and_bindings(
            config_files=[path], bindings=params.get('GIN_CONFIG', None)
        )
        EXPERIMENT.log_parameters(flatten_gin_config(gin.config._CONFIG))
        train()
