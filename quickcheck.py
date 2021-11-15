import torch as t
import numpy as np
from torchtyping import TensorType
FLOAT_TOLERANCE=1e-4

def generate_instance(type_obj):
    if isinstance(type_obj, TensorType):
        print("tensor type")
def check_equality(a,b):
    