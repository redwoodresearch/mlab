from typing import Iterable, Tuple

import gin
import torch as t


class MLABOptim:
    pass


@gin.configurable
class SGD(MLABOptim):
    def __init__(
        self,
        params: Iterable[t.Tensor],
        lr: float,
        momentum: float,
        dampening: float,
        weight_decay: float,
    ):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay

        self.num_steps_taken = 0
        self.velocities = []

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with t.no_grad():
            for i, p in enumerate(self.params):
                g = p.grad.clone()

                if self.weight_decay != 0:
                    g += self.weight_decay * p

                if self.momentum != 0:
                    if self.num_steps_taken > 0:
                        self.velocities[i] = (
                            self.momentum * self.velocities[i]
                            + (1 - self.dampening) * g
                        )
                    else:
                        self.velocities.append(g)

                    g = self.velocities[i]

                p -= self.lr * g

        self.num_steps_taken += 1


@gin.configurable
class RMSProp(MLABOptim):
    def __init__(
        self,
        params: Iterable[t.Tensor],
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.vs = [t.zeros_like(p) for p in self.params]
        self.bs = [t.zeros_like(p) for p in self.params]
        self.gas = [t.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with t.no_grad():
            for i, p in enumerate(self.params):
                g = p.grad.clone()

                g += self.weight_decay * p

                self.vs[i] = self.alpha * self.vs[i] + (1 - self.alpha) * g * g

                self.bs[i] = self.momentum * self.bs[i] + g / (
                    t.sqrt(self.vs[i]) + self.eps
                )
                p -= self.lr * self.bs[i]


@gin.configurable
class Adam(MLABOptim):
    def __init__(
        self,
        params: Iterable[t.Tensor],
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.ms = [t.zeros_like(p) for p in self.params]
        self.vs = [t.zeros_like(p) for p in self.params]

        self.num_steps_taken = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.num_steps_taken += 1
        with t.no_grad():
            for i, p in enumerate(self.params):
                g = p.grad.clone()

                g += self.weight_decay * p

                self.ms[i] = self.betas[0] * self.ms[i] + (1 - self.betas[0]) * g

                self.vs[i] = self.betas[1] * self.vs[i] + (1 - self.betas[1]) * g * g
                n = self.num_steps_taken
                m_scaled = self.ms[i] / (1 - self.betas[0] ** n)
                v_scaled = self.vs[i] / (1 - self.betas[1] ** n)

                p -= self.lr * m_scaled / (t.sqrt(v_scaled) + self.eps)
