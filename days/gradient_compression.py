# reimplementation of https://arxiv.org/pdf/1905.13727.pdf

# you want to approximage an n*m matrix as (n*r) @ (r*m). You could do this with SVD (svd produces orthonormal scale orthonormal, you can take only the top k*x part of the scale matrix, and merge scale into orthonormal, to get low rank representation)
# power sgd has a momentum-like component (it uses an iterative approximation of the low rank decomposition starting with the previous decomposition)

import torch
import torch as t
import torch.distributed as dist


class LowRankCompressionDistributedSGD:
    def __init__(
        self,
        params,
        compression_rank,
        lr: float,
        momentum: float,
    ):
        self.cr = compression_rank
        self.lr = lr
        self.mu = momentum

        params = list(params)
        self.compressed_params = [
            p.view(-1, p.shape[-1]) for p in params if len(p.shape) > 1
        ]
        self.raw_params = [p for p in params if len(p.shape) <= 1]
        self.params = params

        self.momentum_compressed = [torch.zeros_like(p) for p in self.compressed_params]
        self.momentum_raw = [torch.zeros_like(p) for p in self.raw_params]
        self.errors = [torch.zeros_like(p) for p in self.compressed_params]
        # why is this variance 1??? is it???
        self.qs = [torch.randn(p.shape[-1], self.cr) for p in self.compressed_params]
        self.ps = [torch.zeros(p.shape[0], self.cr) for p in self.compressed_params]

    def _all_reduce(self):
        for reduction in [
            dist.all_reduce(p, async_op=True)
            for p in (self.ps + self.qs + [x.grad for x in self.raw_params])
        ]:
            reduction.wait()

    def decompress(self):
        for param, q, p in zip(self.compressed_params, self.qs, self.ps):
            t.matmul(q, p, out=param.grad)

    def add_error(self):
        for param, error in zip(self.compressed_params, self.errors):
            param.grad.add_(error)

    def orthonormalize(matrix):
        for column in range(matrix.shape[1]):
            for pre_column in range(column):
                matrix[:, column] -= (
                    pre_column
                    * t.einsum("i,i->", matrix[:, column], matrix[:, pre_column])
                    / matrix[:, pre_column].norm()
                )
            matrix[:, column] = matrix[:, column].normalize()

    def compress_gradients_and_store_error(self):
        for i, (param, q, p) in enumerate(
            zip(self.compressed_params, self.qs, self.ps)
        ):
            grad = param.grad
            t.matmul(grad.mean(dim=0), q, out=p)
            self.orthonormalize(p)
            t.matmul(grad.permute(1, 0), p, out=q)

            decompressed = t.einsum("nr, mr -> nm", p, q)
            self.errors[i] = grad - decompressed

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for param in self.params:
            assert param.grad is not None
        with torch.no_grad():
            self.add_error()
            self.compress_gradients_and_store_error()
            self._all_reduce()
            self.decompress()

            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad
                if self.b[i] is not None:
                    self.b[i] = self.mu * self.b[i] + g
                else:
                    self.b[i] = g
                g = self.b[i]
                p -= self.lr * g
