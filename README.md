# Double Descent (MLX)

<p align="center">
  <img src="media/polynomial_ols_darkmode.png#gh-dark-mode-only" alt="dd" width="50%">
</p>
<p align="center">
  <img src="media/polynomial_ols.png#gh-light-mode-only" alt="dd" width="50%">
</p>

_Double Descent_: ...

## Background

...

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

