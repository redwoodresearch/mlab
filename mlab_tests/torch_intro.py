import torch as t
from test_all import allclose
import days.torch_intro as reference


def test_weekly_average_temperature(fn):
    input_1 = t.Tensor([71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83])
    input_2 = t.empty(70).normal_(100, 100)
    allclose(fn(input_1), reference.weekly_average_temperature(input_1))
    allclose(fn(input_2), reference.weekly_average_temperature(input_2))
