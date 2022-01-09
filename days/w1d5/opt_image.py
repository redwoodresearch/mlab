import dataclasses
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm

import color
import p_image


@dataclasses.dataclass(frozen=True)
class OptImageResult:
    init_image: t.Tensor
    final_image: t.Tensor
    losses: np.ndarray


def optimize_image(
    img_class,
    loss_fn: Optional[Callable[[t.Tensor], t.Tensor]],
    steps: int = 10,
    transform: bool = True,
    n_transforms: int = 100,
    transforms_list: Optional[List] = None,
    lr: float = 1e-1,
    disable_tqdm: bool = False,
):

    if transforms_list is None:
        transforms_list = [
            transforms.Pad(20, padding_mode="reflect"),
            transforms.RandomAffine(
                degrees=5, translate=(0.05, 0.05), scale=(1, 1.2), shear=5
            ),
            transforms.RandomPerspective(distortion_scale=0.1),
            transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
            ),
            transforms.CenterCrop(224),
        ]

    image_module = img_class().cuda()

    init_image = image_module().clone()
    optimizer = optim.Adam(image_module.parameters(), lr=lr)

    losses = [loss_fn(init_image).mean()]
    for _ in tqdm(range(steps), disable=disable_tqdm):
        optimizer.zero_grad()
        image = image_module()

        if transform:
            batch = t.cat([image_module() for _ in range(n_transforms)], dim=0)

            # p_image.imshow(batch[:1])
            # plt.show();

            for tr in transforms_list:
                batch = tr(batch)

            # p_image.imshow(batch[:1])
            # plt.show();
            # break

        else:
            batch = image_module()

        loss = -loss_fn(batch).mean()
        loss.backward()
        optimizer.step()

        losses.append(float(loss))

    return OptImageResult(
        final_image=image,
        init_image=init_image,
        losses=np.array(losses),
    )


def viz_opt_result(res: OptImageResult):
    ax = plt.subplot(2, 2, 1)
    ax.set_title(f"Original: loss={-res.losses[0]:.3f}")
    p_image.imshow(res.init_image, interpolation="nearest")

    ax = plt.subplot(2, 2, 2)
    ax.set_title(f"Final: loss={-res.losses[-1]:.3f}")
    p_image.imshow(res.final_image, interpolation="nearest")

    ax = plt.subplot(2, 2, 3)
    init_hsv = color.rgb_to_hsv(res.init_image)
    plt.hist(
        init_hsv[:, 0].flatten().detach().cpu().numpy() / 2 / np.pi,
        label="h",
        bins=128,
        alpha=0.5,
    )
    plt.hist(
        init_hsv[:, 1].flatten().detach().cpu().numpy(),
        label="s",
        bins=128,
        alpha=0.5,
    )
    plt.hist(
        init_hsv[:, 2].flatten().detach().cpu().numpy(),
        label="v",
        bins=128,
        alpha=0.5,
    )
    plt.legend()

    ax = plt.subplot(2, 2, 4)
    final_hsv = color.rgb_to_hsv(res.final_image)
    plt.hist(
        final_hsv[:, 0].flatten().detach().cpu().numpy() / 2 / np.pi,
        label="h",
        bins=128,
        alpha=0.5,
    )
    plt.hist(
        final_hsv[:, 1].flatten().detach().cpu().numpy(),
        label="s",
        bins=128,
        alpha=0.5,
    )
    plt.hist(
        final_hsv[:, 2].flatten().detach().cpu().numpy(),
        label="v",
        bins=128,
        alpha=0.5,
    )
    plt.legend()


def get_model_channel_output(model: nn.Module, layer_str: str, input: t.Tensor):
    saved_output = None

    def hook(_, __, output):
        nonlocal saved_output
        saved_output = output

    module = model
    for name in layer_str.split("."):
        module = getattr(module, name)
    module.register_forward_hook(hook)

    model(input)

    saved_output = t.mean(F.relu(saved_output), dim=[-2, -1])
    return saved_output


def get_model_output(model: nn.Module, layer_str: str, input: t.Tensor):
    saved_output = None

    def hook(_, __, output):
        nonlocal saved_output
        saved_output = output

    module = model
    for name in layer_str.split("."):
        module = getattr(module, name)
    module.register_forward_hook(hook)

    model(input)

    return saved_output
