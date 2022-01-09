import gin


@gin.configurable
def run(lr, zf):
    print("lr", lr)
    print("zf", zf)
