from hpsearch import hpsearch
from datetime import datetime

tag = "tuning_learning_rate_"
tag += datetime.now().strftime("%Y%m%d%H%M%S")

hpsearch(
    "bert_finetuning",
    "days.w2d4.bert_finetuning.train",
    "days/w2d4/bert_finetuning.gin",
    {"train.lr": [1e-4, 5e-5, 1e-5, 5e-6, 1e-6], "set_random_seed.seed":[0,1,2,3], "train.tag": [tag]},
    # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
    comet_key="EVwneUg4V62GWa4Vpu1JcjzpF",
    local = False # set to false after it works locally! 
)

