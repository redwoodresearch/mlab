
import sys
import os
sys.path.append("../../")
from hpsearch import hpsearch
os.chdir("/home/ubuntu/mlab")

hpsearch(
    "nikola_rudolf_mlps_without_activation",
    "days.w2d4.demo_train.train",
    "days/w2d4/demo_train.gin",
    {
        "train.lr": [1e-3, 1e-4, 1e-5],
        "MyModel.hidden_size": [32, 64],
        "set_random_seed.seed":[0,1,2,3],
        "MyModel.input_size": [2],
        "MyModel.output_size": [2],
        "MyModel.n_layers": [2],
        "train.batch_size": [6],
        "train.num_epochs": [1]
    },
    # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
    comet_key="c0KCCBHfreDu3cCto8zVZ1G6f",
    local = True # set to false after it works locally! 
)
