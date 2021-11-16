import torch as t
import numpy as np


def tstat(name, tensor):
    print(
        name, "mean", "{0:.4g}".format(t.mean(tensor).cpu().item()), "var", "{0:.4g}".format(t.var(tensor).cpu().item())
    )


def tpeek(name, tensor):
    print(
        f"{name} MEAN: {'{0:.4g}'.format(t.mean(tensor).cpu().item())} VAR: {'{0:.4g}'.format(t.var(tensor).cpu().item())} SHAPE {tuple(tensor.shape)} VALS [{' '.join(['{0:.4g}'.format(x) for x in t.flatten(tensor)[:10].cpu().tolist()])}]"
    )
