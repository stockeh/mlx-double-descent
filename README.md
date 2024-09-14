# Double Descent (MLX)

<p align="center">
  <img src="media/polynomial_ols_darkmode.png#gh-dark-mode-only" alt="dd" width="50%">
</p>
<p align="center">
  <img src="media/polynomial_ols.png#gh-light-mode-only" alt="dd" width="50%">
</p>

_Double Descent_: a phenomenon in machine learning where test error initially increases near the interpolation threshold, as model parameters approach the number of samples, then decreases and improves generalization in the highly overparameterized regime.​ The term comes from [Belkin et al. (2019)](https://arxiv.org/abs/1812.11118) and this repo is inspired by [Schaeffer et al. (2023)](https://arxiv.org/abs/2303.14151).

## Background

We consider the case of modeling $f: x \mapsto y$ where $y = 2x + \cos \left(25x / \sin x\right)$ using polynomial regression. While we could solve this with gradient descent, optimized with SGD or Adam, we observe double descent consistently with a the Moore–Penrose inverse (pseudoinverse) solution to ordinary least squares.

## Running

Run with default params and save the result in [`media/polynomial_*.png`](media/polynomial_ols.png):
```bash
python polynomial.py
```
- **`polynomial.py`**: training and evaluation loops
- **`optimizers.py`**: sgd, adam, ols (pseudoinverse), ...
- **`metrics.py`**: mse, rmse, r2, ...
- **`data.py`**: generate the dataset

## Dependencies

Install the dependencies (optimized for Apple silicon; yay for [MLX](https://github.com/ml-explore/mlx)!):
```bash
pip install -r requirements.txt
```

