import hpsearch
import pathlib
import os

GIT_ROOT = pathlib.Path(__file__).absolute().parent.parent.parent
os.chdir(GIT_ROOT)
API_KEY = "qjxcybqq2HsGHbEwATgNiqWgE"  # Tony Wang's key


# hpsearch.hpsearch(
#     "arthur-tony-test-0",
#     "days.w2d4.demo_train.train",
#     "days/w2d4/demo_train.gin",
#     {
#         "lr": [1e-3, 1e-4, 1e-5],
#         # "MyModel.hidden_size": [32, 64],
#         # "set_random_seed.seed": [0, 1, 2, 3],
#     },
#     # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
#     comet_key=API_KEY,
#     local=False,  # set to false after it works locally!
# )

hpsearch.hpsearch(
    "arthur-tony-bert-finetune-1",
    "days.w2d4.bert_finetune.train",
    "days/w2d4/bert_finetune.gin",
    {
        "lr": [1e-3, 1e-4, 1e-5],
        # "MyModel.hidden_size": [32, 64],
        # "set_random_seed.seed": [0, 1, 2, 3],
    },
    # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
    comet_key=API_KEY,
    local=True,  # set to false after it works locally!
)
