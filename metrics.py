import mlx.core as mx


def mse(T, Y):
    return mx.mean((T - Y)**2).item()


def rmse(T, Y):
    return mx.sqrt(mx.mean((T - Y)**2)).item()


def r2(T, Y):
    ss_res = mx.sum((T - Y) ** 2)
    ss_tot = mx.sum((T - mx.mean(T)) ** 2)
    return (1 - (ss_res / ss_tot)).item()
