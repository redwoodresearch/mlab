import days.bert as bert
import torch
import torch as t
import inspect
from pythonfuzz.main import PythonFuzz


def getmembers(module):
    return inspect.getmembers(module, lambda x: inspect.isfunction(x) or inspect.isclass(x))


def get_fn_of_shape(shape):
    return t.random.uniform(-1, 1, shape)


def test_reimplementation_module(target, references):
    target_exported = {k: v for k, v in getmembers(target)}
    references_exported = {pair[0]: pair[1] for module in references for pair in getmembers(module)}
    for k, target in target_exported.items():
        if k in references_exported:
            print("matching key", k)
            reference = references_exported[k]
            if inspect.isfunction(target):

                @PythonFuzz
                def fuzzTheseTwo():
                    pass

                fuzzTheseTwo()
            elif inspect.isclass(target):
                print("cant test classes yet")
    print(target_exported)


if __name__ == "__main__":
    test_reimplementation_module(bert, [torch, torch.nn])
