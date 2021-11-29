import mlab_tests.bert
import mlab_tests.nn_functional
import mlab_tests.torch_intro


def test_fn(**kwargs):
    for arg, val in kwargs.items():
        print(arg, val)
