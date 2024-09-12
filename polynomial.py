import argparse
import mlx.core as mx
import matplotlib.pyplot as plt

from tqdm import tqdm
from metrics import mse
from optimizers import adam, sgd, ols
from data import generate, legendre, vander

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument('-e', '--epochs', type=int,
                    default=50, help='number of epochs')
parser.add_argument('--optim', type=str, default='ols',
                    choices=['adam', 'sgd', 'ols'], help='optimizer')
parser.add_argument('--basis', type=str, default='legendre',
                    choices=['legendre', 'vander'], help='polynomial basis')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--cpu', action='store_true', help='use cpu only')

###############################################################################

N_SAMPELS = 15
DEGREES = mx.arange(1, 26).tolist() + [30, 40, 50, 100]
REPEAT = 10

###############################################################################


def forward(X, W):
    return X @ W


def gradient_f(X, T, W):
    Y = forward(X, W)
    dEdY = -2 * (T - Y)
    dYdW = X
    return dYdW.T @ dEdY / X.shape[0]


def main(args):
    mx.random.seed(args.seed)
    Xtest, Ttest = generate(-1, 1, 1000, train=False)
    train_error, test_error = [], []

    basis = legendre if args.basis == 'legendre' else vander

    for n in tqdm(DEGREES, desc='Processing Degrees'):
        trials_train_error, trials_test_error = [], []
        for _ in tqdm(range(REPEAT), desc='Repeating', leave=False):
            Xtrain, Ttrain = generate(-1, 1, N_SAMPELS, train=True)
            XtrainP = basis(Xtrain, n)
            XtestP = basis(Xtest, n)

            if args.optim == 'sgd':
                W = mx.zeros((n, 1))
                W, _, _ = sgd(XtrainP, Ttrain, W,
                              forward, gradient_f,
                              args.lr, args.epochs)
            elif args.optim == 'adam':
                W = mx.zeros((n, 1))
                W, _, _ = adam(XtrainP, Ttrain, W,
                               forward, gradient_f,
                               args.lr, args.epochs)
            elif args.optim == 'ols':
                W = ols(XtrainP, Ttrain, pseudoinverse=True)
            else:
                raise ValueError('Invalid optimizer')

            trials_train_error.append(mse(Ttrain, forward(XtrainP, W)))
            trials_test_error.append(mse(Ttest, forward(XtestP, W)))

        train_mean = mx.mean(mx.array(trials_train_error)).item()
        test_mean = mx.mean(mx.array(trials_test_error)).item()

        train_error.append((train_mean, 1.96 * (mx.std(mx.array(trials_train_error),
                                                       ddof=1) / mx.sqrt(REPEAT))))
        test_error.append((test_mean, 1.96 * (mx.std(mx.array(trials_test_error),
                                                     ddof=1) / mx.sqrt(REPEAT))))

    train_mean = mx.array([te[0] for te in train_error])
    train_ci = mx.array([te[1] for te in train_error])
    test_mean = mx.array([te[0] for te in test_error])
    test_ci = mx.array([te[1] for te in test_error])

    # !plotting
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(DEGREES, train_mean, '.-', label='Train', lw=2, color='#1b9e77')
    ax.fill_between(DEGREES, train_mean - train_ci,
                    train_mean + train_ci, color='#1b9e77', alpha=0.2)
    ax.plot(DEGREES, test_mean, '.-', label='Val', lw=2, color='#d95f02')
    ax.fill_between(DEGREES, test_mean - test_ci,
                    test_mean + test_ci, color='#d95f02', alpha=0.2)
    interpolation = ax.axvline(N_SAMPELS, color='k', linestyle='--',
                               label='Interpolation', lw=2)

    ax.set_title(f'Polynomial Regression \
                 {args.optim.upper()} ($n={{{REPEAT}}}$)')
    ax.set_xlabel('Num. Parameters (Num Features)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_ylim(bottom=1e-3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    legend = ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(f'media/polynomial_{args.optim}.png',
                dpi=300, bbox_inches='tight')

    plt.show()

    # dark mode
    interpolation.set_color('white')
    legend = ax.legend(loc='upper left', frameon=False)
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(axis='both', colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    plt.setp(legend.get_texts(), color='white')
    fig.tight_layout()
    fig.savefig(f'media/polynomial_{args.optim}_darkmode.png',
                dpi=300, transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
