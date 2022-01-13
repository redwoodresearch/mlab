import sys
sys.path.append("/home/p/Desktop/Programs/dataspell")
print(sys.path)
from mlab.hpsearch import hpsearch
def main():
    hpsearch(
        "mlps_without_activations",
        "days.w2d4.demo_train.train",
        "days/w2d4/demo_train.gin",
        {"lr": [1e-3, 1e-4, 1e-5], "MyModel.hidden_size": [32, 64],'set_random_seed.seed':[0,1,2,3]},
        # We provide an additional Gin-configurable function set_random_seed(seed). This means you can control it in your gin without writing the function yourself
        comet_key="Ch9OIxODMWCZuK2LSscvbculp",
        local=True # set to false after it works locally!
    )


if __name__ == '__main__':
    main()