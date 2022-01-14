from hpsearch import hpsearch
hpsearch(
    "mlps_without_activations",
    "days.w2d4.bert_train.train",
    "days/w2d4/bert_train.gin",
    {
        "train.lr": [1e-4, 1e-5, 1e-6], 
        "train.max_seq_len": [128, 512], 
        "train.batch_size": [16], 
        "set_random_seed.seed": [0]
    },
    comet_key="vaznwXsdK5Z3Hug3FKZCl9lGN",
    local = False
)
