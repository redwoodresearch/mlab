import os
from hpsearch import hpsearch


os.chdir("../..")


api_key = "OmfvOU0RmHbt4iMa2WIYGBjBf"

hpsearch(
    "inappropriate ioctl",
    "days.w2d4.demo_train.train",
    "days/w2d4/demo_train.gin",
    {"train.lr": [1e-3, 1e-4, 1e-5], "MyModel.hidden_size": [32, 64], "set_random_seed.seed":[0,1,2,3]},
    # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
    comet_key=api_key,
    local = True # set to false after it works locally! 
)

# "set_random_seed.seed":[0,1,2,3]