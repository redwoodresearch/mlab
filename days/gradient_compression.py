# reimplementation of https://arxiv.org/pdf/1905.13727.pdf

# you want to approximage an n*m matrix as (n*r) @ (r*m). You could do this with SVD (svd produces orthonormal scale orthonormal, you can take only the top k*x part of the scale matrix, and merge scale into orthonormal, to get low rank representation)
# power sgd has a momentum-like component (it uses an iterative approximation of the low rank decomposition starting with the previous decomposition)

import torch
import torch as t
import torch.distributed as dist
import torch.nn.functional as F
from typing import *


class LowRankCompressionDistributedSGD:
    def __init__(
        self,
        params: List[t.nn.Parameter],
        compression_rank: int,
        dist_size,
        lr: float,
        momentum: float,
    ):
        self.cr = compression_rank
        self.lr = lr / dist_size
        self.mu = momentum
        params = [param for param in params if param.requires_grad]
        self.device = params[0].device
        # all compressed parameters have to be rank 2, for conv "output (input h w)" works best, but ideally it'd be "largest (all others)"
        self.compressed_params = [
            p.view(p.shape[0], -1).to(self.device) for p in params if len(p.shape) > 1
        ]
        self.raw_params = [p for p in params if len(p.shape) <= 1]
        self.params = params

        self.momentum = [None for p in self.params]
        self.errors = [
            torch.zeros_like(p).to(self.device) for p in self.compressed_params
        ]
        # why is this variance 1??? is it???
        self.qs = [
            torch.randn(p.shape[-1], self.cr).to(self.device)
            for p in self.compressed_params
        ]
        self.ps = [
            torch.zeros(p.shape[0], self.cr).to(self.device)
            for p in self.compressed_params
        ]

    def share_raw_grads(self):
        for reduction in [
            dist.all_reduce(p.grad, async_op=True) for p in self.raw_params
        ]:
            reduction.wait()

    def decompress(self):
        for param, q, p in zip(self.compressed_params, self.qs, self.ps):
            t.matmul(p, q.permute(1, 0), out=param.grad)

    def add_error(self):
        for param, error in zip(self.compressed_params, self.errors):
            param.grad.add_(error)

    def orthonormalize(self, matrix):
        for column in range(matrix.shape[1]):
            for pre_column in range(column):
                matrix[:, column] -= (
                    pre_column
                    * t.einsum("i,i->", matrix[:, column], matrix[:, pre_column])
                    / matrix[:, pre_column].norm()
                )
            matrix[:, column] = F.normalize(matrix[:, column], dim=0)

    def compress_gradients_and_store_error(self):
        reductions = []
        for i, (param, q, p) in enumerate(
            zip(self.compressed_params, self.qs, self.ps)
        ):
            t.matmul(param.grad, q, out=p)
            reductions.append(dist.all_reduce(p, async_op=True))
        for reduction in reductions:
            reduction.wait()

        reductions = []
        for i, (param, q, p) in enumerate(
            zip(self.compressed_params, self.qs, self.ps)
        ):
            self.orthonormalize(p)
            t.matmul(param.grad.permute(1, 0), p, out=q)

            decompressed = t.einsum("nr, mr -> nm", p, q)
            self.errors[i] = param.grad - decompressed
            reductions.append(dist.all_reduce(q, async_op=True))

        for reduction in reductions:
            reduction.wait()

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for param in self.params:
            assert isinstance(param.grad, t.Tensor)
        with torch.no_grad():
            # setting grad of my views on parameters because they don't automatically get grad
            for param, cp in zip(
                [x for x in self.params if len(x.shape) > 1], self.compressed_params
            ):
                cp.grad = param.grad.view(param.grad.shape[0], -1)

            self.add_error()
            self.compress_gradients_and_store_error()
            self.share_raw_grads()
            self.decompress()

            for i, p in enumerate(self.params):
                g = p.grad
                if self.momentum[i] is not None:
                    self.momentum[i] = self.mu * self.momentum[i] + g
                else:
                    self.momentum[i] = g
                g = self.momentum[i]
                p -= self.lr * g
