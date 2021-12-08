import gin
from comet_ml import Experiment
import random
import torch as t
from days.search_jobs import Search


@gin.configurable()
def load_data():
    pass

@gin.configurable()
class Model():
    def __init__(self, hidden_size):

@gin.configurable()
def train(num_steps=10, lr=0.1):
    model = Model()
    for i in range(num_steps):
        loss = random.randint(0, 30)
        ts = t.nn.Parameter(t.rand(10, 10))
        loss = ts.sum()
        loss.backward()

if __name__=="__main__":
    gin.load_configs_and_bindings([], {"train.num_steps":1,"train.lr":0.01,"model.hidden_size":1024})
    
    for i in range(10):
        # Create an experiment with your api key
        experiment = Experiment(
            api_key="K9WmPOGXBluMVkA5hFlz0P0dL",
            project_name="many_experiment_per_process_2",
            workspace="orionarmsinger",
        )

        experiment.log_asset

        train()
        # mysearch = Search(
        #     function_path="days.hpsearch.train",
        #     grid={
        #         "train.num_steps": range(10, 100, 30),
        #     },
        # )
        # mysearch.orchestrate()
