from hpsearch import hpsearch

hpsearch(
    "mlps_without_activations_warren_tim",
    "days.w2d4.demo_train.train",
    "days/w2d4/demo_train.gin",
    {
        # "lr": [1e-3, 1e-4, 1e-5],
        "lr": [1e-3],
        "MyModel.hidden_size": [32, 64],
        "set_random_seed.seed": [0, 1, 2, 3],
    },
    # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
    comet_key="UKJWpl2psmEN0DBq0vWBXW5rO",
    local=False,  # set to false after it works locally!
)
