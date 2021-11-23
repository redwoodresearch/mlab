import torch as t
from einops import rearrange, reduce, repeat


def average_weekly_temperature(daily_temperature):
    return reduce(daily_temperature, "(weeks days) -> weeks", "mean", days=7)


def normalize_temperature(daily_temperature):
    mean_normalized = daily_temperature - repeat(
        reduce(daily_temperature, "(weeks days) -> weeks", "mean", days=7), "weeks -> (weeks 7)"
    )
    var_normalized = mean_normalized / repeat(
        reduce(mean_normalized ** 2, "(weeks days) -> weeks", "mean", days=7) ** 0.5, "weeks -> (weeks 7)"
    )
    return var_normalized
