from hpsearch import hpsearch
from datetime import datetime

tag = "batch_size_seq_len_"
tag += datetime.now().strftime("%Y%m%d%H%M%S")

hpsearch(
    "bert_finetuning",
    "days.w2d4.bert_finetuning.train",
    "days/w2d4/bert_finetuning.gin",
    {"train.lr": [3e-5, 1e-5, 3e-6], "train.batch_size": [4, 8, 16], "set_random_seed.seed":[0], "train.max_seq_len": [256, 384, 512], "train.tag": [tag]},
    # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
    comet_key="EVwneUg4V62GWa4Vpu1JcjzpF",
    local = False # set to false after it works locally! 
)

