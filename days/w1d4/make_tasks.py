def make_grid(possible_values):
    size = reduce(mul, map(len, possible_values.values()))
    ls = []
    for idx in range(size):
        new_values = {}
        for name, values in possible_values.items():
            new_values[name] = values[idx % len(values)]
            idx //= len(values)
        ls.append({"priority": 1, "parameters": new_values})
    return ls

def thangify(pair):
    name, value = pair
    return f"{name}={value}"

def stringify(dictionary):
    return list(map(thangify, list(dictionary.items())))

# possible vals
possible_values = {
    "gin_train.lr": [0.01],
    "Model.P": [2],
    "Model.H": [100],
    "Model.K": [3],
    "gin_train.betas": [(0.9, 0.999)]
}