from typing import Any, Callable

import torch as t
from torch import nn
from torch.utils.hooks import RemovableHandle


class HookHandler:
    def __init__(self):
        self.activations = {}
        self.hook_handles: list[RemovableHandle] = []

    def reset(self):
        for h in self.hook_handles:
            h.remove()

        self.activations = {}
        self.hook_handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reset()
        # print("All hooks removed!")

    def add_hook(
        self,
        mod: nn.Module,
        fn: Callable[[nn.Module, Any, t.Tensor], Any],
    ):
        self.hook_handles.append(mod.register_forward_hook(fn))

    def add_save_activation_hook(
        self,
        mod: nn.Module,
        key: str,
    ):
        def fn(model, input, output):
            self.activations[key] = output.detach()

        self.hook_handles.append(mod.register_forward_hook(fn))
