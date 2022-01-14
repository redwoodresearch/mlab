from hpsearch import hpsearch
hpsearch(
    "nat_eric_test",
    "bert_eric.train",
    "bert_sol_gin.gin",
    {"lr": [1e-3, 1e-4],
     #"Bert.config['hidden_size']": [32, 64],
     "set_random_seed.seed":[1,2]},
    # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
    comet_key="XcDtpxPXqwZXYhcjjJsBGmp3P",
    local = True # set to false after it works locally! 
)