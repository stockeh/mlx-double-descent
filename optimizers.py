import mlx.core as mx

from metrics import rmse


def sgd(X, T, W, forward, gradient_f, eta, epochs):
    error_trace = []
    W_trace = []
    for _ in range(epochs):
        W_trace.append(mx.flatten(W))
        W -= eta * gradient_f(X, T, W)
        error_trace.append(rmse(T, forward(X, W)))
    return W, error_trace, W_trace


def adam(X, T, W, forward, gradient_f, eta, epochs, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m, v = 0, 0
    error_trace, W_trace = [], []
    for step in range(epochs):
        W_trace.append(W.flatten())
        g = gradient_f(X, T, W)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mhat = m / (1 - beta1 ** (step+1))
        vhat = v / (1 - beta2 ** (step+1))
        W -= eta * mhat / (mx.sqrt(vhat) + epsilon)
        error_trace.append(rmse(T, forward(X, W)))
    return W, error_trace, W_trace


def ols(X, T, pseudoinverse=True):
    if pseudoinverse:  # Moore-Penrose pseudoinverse
        U, S, Vt = mx.linalg.svd(X, stream=mx.cpu)
        K = min(X.shape)
        U, Vt = U[:, :K], Vt[:K, :]
        s_inv = mx.where(S > 1e-15, 1 / S, 0)
        pinv = Vt.T @ (mx.diag(s_inv) @ U.T)
        return pinv @ T
    else:  # direct inverse (must be full rank)
        return mx.linalg.inv(X.T @ X, stream=mx.cpu) @ X.T @ T
