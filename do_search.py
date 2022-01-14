from days.w2d1.bert_sol import BertWithClassify
from hpsearch import hpsearch

hpsearch(
    name="jenny-guilhermo-bert-sentiment",
    fn_path="days.w2d4.train.train",
    base_config="/home/ubuntu/mlab/days/w2d4/bert.gin",
    search_spec={"train.lr": [1e-3, 1e-4, 1e-5],"set_random_seed.seed":[0,1,2], "train.max_len": [64, 128, 256, 512]},
    # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
    comet_key="OiNBEOeeT9IFDdHDHRLeEe5hb",
    local = False # set to false after it works locally! 
)