from typing import Any, Callable, List, Tuple

import torch as t
from torch import nn
from torch.utils.hooks import RemovableHandle


class HookHandler:
    def __init__(self):
        self.activations = {}
        self.grads = {}
        self.hook_handles: List[RemovableHandle] = []

    def reset(self):
        for h in self.hook_handles:
            h.remove()

        self.activations = {}
        self.grads = {}
        self.hook_handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reset()
        print("All hooks removed!")

    def add_save_activation_hook(
        self,
        mod: nn.Module,
        key: str,
    ):
        def fn(model, input, output):
            self.activations[key] = output.detach()

        self.hook_handles.append(mod.register_forward_hook(fn))

    def add_save_grad_hook(
        self,
        mod: nn.Module,
        key: str,
    ):
        def fn(model, grad_input, grad_output):
            self.grads[key] = (grad_input, grad_output)

        self.hook_handles.append(mod.register_backward_hook(fn))

    def add_save_grad_norm_hook(
        self,
        mod: nn.Module,
        key: str,
    ):
        def fn(model, grad_input, grad_output):
            self.grads[key] = (
                t.linalg.norm(grad_input[0], dim=-1),
                t.linalg.norm(grad_output[0], dim=-1),
            )

        self.hook_handles.append(mod.register_full_backward_hook(fn))

    def add_backward_hook(
        self,
        mod: nn.Module,
        fn: Callable[[nn.Module, Tuple[t.Tensor, ...], Tuple[t.Tensor, ...]], Any],
    ):
        self.hook_handles.append(mod.register_backward_hook(fn))


trivial_induction_head_hypothesis = (
    t.cat(
        (
            t.tensor([0.0], device=tokens.device),
            tokens[:-1],
        )
    ).unsqueeze(1)
    == tokens.unsqueeze(0)
)
