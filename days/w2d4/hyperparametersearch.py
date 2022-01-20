from hpsearch import hpsearch

# hpsearch(
#     name="lukas_daniel_test_reallyremote",
#     fn_path="days.w2d4.demo_train.train",
#     base_config="days/w2d4/demo_train.gin",
#     # search_spec={"lr": [1e-3, 1e-4, 1e-5], "MyModel.hidden_size": [32, 64],"set_random_seed.seed":[0,1,2,3]},
#     search_spec={"lr": [1e-3], "MyModel.hidden_size": [32, 64],"set_random_seed.seed":[0]},
#     # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
#     comet_key="72XQSdnwnBcob4Q8NpbJHewll",
#     local = False # set to false after it works locally! 
# )
hpsearch(
    name="lukas_daniel_bert_local",
    fn_path="days.w2d4.hpbertfinetuning.run",
    base_config="days/w2d4/train.gin",
    search_spec={"run.lr": [1e-3, 1e-5, 1e-6], "run.seed":[0,1,2]},
    # search_spec={"run.lr": [1e-4], "run.seed":[0]},
    # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
    comet_key="72XQSdnwnBcob4Q8NpbJHewll",
    local = True # set to false after it works locally! 
)
https://www.comet.ml/docs/user-interface/models/
experiment.log_model("MNIST CNN", "../models/run-026")