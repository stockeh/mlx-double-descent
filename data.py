import mlx.core as mx
import matplotlib.pyplot as plt


def legendre(x: mx.array, n: int) -> mx.array:
    """Legendre polynomial features. Ignores P_0(x) = 1.0."""
    o = mx.zeros((x.shape[0], n))
    o[:, 0] = x[:, 0]  # P_1(x) = x

    if n > 1:  # P_n(x) for n >= 2 via recurrence
        for i in range(1, n):
            if i == 1:
                o[:, i] = (3 * x[:, 0] * o[:, i-1]) / 2 - 0.5
            else:
                o[:, i] = ((2 * i + 1) * x[:, 0] * o[:, i-1] -
                           i * o[:, i-2]) / (i + 1)
    return o


def vander(x: mx.array, n: int) -> mx.array:
    """Vandermonde matrix."""
    return mx.power(x, mx.array([i for i in range(1, n+1)]))


def f(X: mx.array) -> mx.array:
    return mx.add(2.0 * X, mx.cos(X * 25 / mx.sin(X)))


def generate(lo: int, hi: int, n: int, train: bool = False) -> tuple:
    X = mx.random.uniform(lo, hi, shape=(n, 1)) \
        if train else mx.linspace(lo, hi, n).reshape(-1, 1)
    T = f(X)
    return X, T


if __name__ == '__main__':
    X, T = generate(-1, 1, 1000, train=True)
    n = 4
    Xv = vander(X, n)
    Xl = legendre(X, n)

    # !plotting
    inds = mx.argsort(mx.flatten(X))

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    for i in range(n):
        ax[0].plot(X[inds], Xv[inds, i], label=f'$x^{i+1}$')
    ax[0].set_title('Vandermonde Matrix')
    ax[0].legend(ncols=2)

    for i in range(n):
        ax[1].plot(X[inds], Xl[inds, i], label=f'$P_{i+1}(x)$')
    ax[1].set_title('Legendre Polynomial')
    ax[1].legend(ncols=2)

    fig.tight_layout()
    plt.show()
