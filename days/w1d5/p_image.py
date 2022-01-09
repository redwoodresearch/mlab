"""Parameterized images."""
from typing import Tuple, Union

import einops
import numpy as np
import torch as t
import torch.nn as nn

import color

import matplotlib.pyplot as plt


def imshow(img: Union[np.ndarray, t.Tensor], **kwargs):
    if isinstance(img, t.Tensor):
        img = img.cpu()

    with t.no_grad():
        if len(img.shape) == 4:
            img = einops.rearrange(img, "1 ... -> ...")

        if img.shape[0] == 3:
            img = einops.rearrange(img, "c h w -> h w c")

        plt.margins(tight=True)
        plt.imshow(img, **kwargs)
        plt.axis("off")


class NaiveImage(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    ):
        super().__init__()
        self.data: nn.parameter.Parameter = nn.Parameter(t.rand(size=image_shape))

    def forward(self):
        return self.data.clip(min=0, max=1)


class SigmoidImage(NaiveImage):
    def forward(self):
        return t.sigmoid(self.data)


class TiledImage(nn.Module):
    def __init__(
        self,
        ratio: int,
        image_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    ):
        super().__init__()

        self.image_shape = image_shape
        self.tile_shape = (
            image_shape[0],
            image_shape[1],
            image_shape[2] // ratio,
            image_shape[3] // ratio,
        )
        self.data: nn.parameter.Parameter = nn.Parameter(t.randn(size=self.tile_shape))

    def forward(self):
        return t.sigmoid(
            einops.repeat(
                self.data,
                "1 c h w -> 1 c (i h) (j w)",
                i=self.image_shape[2] // self.tile_shape[2],
                j=self.image_shape[3] // self.tile_shape[3],
            )
        )


class FourierImage(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        normalize: bool = True,
        init_std: float = 0.05,
    ):
        super().__init__()
        init = (
            t.randn(
                size=(
                    image_shape[0],
                    image_shape[1],
                    image_shape[2],
                    image_shape[3] // 2 + 1,
                    2,
                ),
            )
            * init_std
        )

        self.normalize = normalize
        if self.normalize:
            H = image_shape[2]
            W = image_shape[3]

            fx = t.fft.fftfreq(W)
            fy = t.fft.fftfreq(H)

            sos = t.zeros(image_shape)
            sos += fx.reshape(1, 1, 1, -1) ** 2
            sos += fy.reshape(1, 1, -1, 1) ** 2

            self.norm_factor: t.Tensor
            self.register_buffer(
                "norm_factor",
                (1 / t.sqrt(sos)[:, :, :, : init.shape[-2]]).unsqueeze(dim=-1),
            )
            self.norm_factor[:, :, 0, 0] = 1

        self.data: nn.parameter.Parameter = nn.Parameter(init)

    def normalize_fn(self, xs: t.Tensor):
        return (xs - xs.mean()) / xs.std()

    def forward(self, ret_raw_logits: bool = False):
        coeffs = self.data
        if self.normalize:
            coeffs = self.data * self.norm_factor

        raw_logits = t.fft.irfft2(t.view_as_complex(coeffs), norm="ortho")
        if ret_raw_logits:
            return raw_logits

        norm_logits = self.normalize_fn(raw_logits)
        return t.sigmoid(norm_logits)


class FourierDecorrImage(nn.Module):

    projection: t.Tensor = (
        t.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]) * 4
    )

    def forward(self):
        decorr = super().forward()
        rgb = t.einsum("i j, b j h w -> b i h w", self.projection, decorr)
        return t.sigmoid(rgb)


class FourierHSVImage(FourierImage):
    def forward(self):
        raw_logits = super().forward(ret_raw_logits=True)

        hsv = t.cat(
            (
                (raw_logits[:, 0:1] * 4) % (2 * t.pi), # H
                t.sigmoid(self.normalize_fn(raw_logits[:, 1:2]) / 2 - 1), # S
                t.sigmoid(self.normalize_fn(raw_logits[:, 2:]) / 2 + 1), # V
            ),
            dim=1,
        )

        return color.hsv_to_rgb(hsv)
