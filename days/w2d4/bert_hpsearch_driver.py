#!/usr/bin/env python3
from hpsearch import hpsearch

def perform(spec):
    hpsearch(
        name="bmillwood",
        fn_path="days.w2d4.bert_hpsearch.main",
        base_config="days/w2d4/bert_train.gin",
        search_spec=spec,
        comet_key="absncaDYNLt6jpNh1Ez0OIVTe",
        local=True,
    )

def defang(spec):
    for v in spec.values():
        del v[1:]
    
spec = {
    "train.lr": [1e-3, 1e-4, 1e-5],
    "train.batch_size": [32, 64],
    "set_random_seed.seed": [0, 1, 2, 3],
}

defang(spec)

perform(spec=spec)