import hpsearch

hpsearch.hpsearch(
    "bert_hp_search",
    "days.w2d4.bert_train.train",
    "days/w2d4/bert_train.gin",
    {"train.lr": [1e-3, 1e-4, 1e-5], "train.seed": [0, 1, 2],},
    comet_key="vABV7zo6pqS7lfzZBhyabU2Xe",
    local=True,
)

# REPLACE COMET API KEY WITH YOUR OWN!
