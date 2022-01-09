import importlib
import search

importlib.reload(search)
search.hpsearch(
    "tao_testies",
    "hpsearch_test.run",
    "hpsearch_test.gin",
    {"run.lr": [0.1, 0.01], "run.zf": [1, 2]},
)
