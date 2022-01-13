import hpsearch

hpsearch.hpsearch(
    "mlps_without_activations",
    "days.w2d4.demo_train.train",
    "days/w2d4/demo_train.gin",
    {"lr": [1e-3, 1e-4, 1e-5], "MyModel.hidden_size": [32, 64]},
    comet_key="vABV7zo6pqS7lfzZBhyabU2Xe",
    local=True,
)
# REPLACE COMET API KEY WITH YOUR OWN!
