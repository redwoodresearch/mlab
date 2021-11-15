import days.bert as bert
import torch
import torch as t
import inspect
import typing
from pythonfuzz.main import PythonFuzz


def getmembers(module):
    return inspect.getmembers(module, lambda x: inspect.isfunction(x) or inspect.isclass(x))


def tensor_of_spec_shape(shape):
    return t.random.uniform(-1, 1, shape)


def test_reimplementation_module(target_module, reference_modules, pairs=[]):
    target_exported = {k: v for k, v in getmembers(target_module)}
    references_exported = {pair[0]: pair[1] for module in reference_modules for pair in getmembers(module)}
    for k, target in target_exported.items():
        # exclude transitive members from other files :)))
        if k in references_exported and inspect.getsourcefile(target) == inspect.getsourcefile(target_module):
            reference = references_exported[k]
            pairs.append(target, reference)
    print("pairs are", pairs)
    for target, reference in pairs:
        if inspect.isfunction(target):
            type_hints = typing.get_type_hints(target)
            print("type hints", type_hints)
            target_argspec = inspect.signature(target)
            print("argspec", target_argspec)

        elif inspect.isclass(target):
            print("cant test classes yet")


if __name__ == "__main__":
    test_reimplementation_module(bert, [torch, torch.nn, torch.nn.functional])
