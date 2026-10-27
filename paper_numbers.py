"""Print the numbers in the paper that dl_obfuscation_v2.ipynb does not print itself.

    python paper_numbers.py results_per_seed.csv results_2026-10-21.csv data

Reads the per-seed results of the ten-seed run, the means of the earlier three-seed run, and the MNIST labels in
data (torchvision downloads them if they are missing). Means of accuracies are computed exactly from the integer
counts and rounded half up, so a tie such as 36.395 prints as 36.40 where the notebook prints 36.39. The part of
the paper that uses each number is given in brackets. The numbers of the two CPU checks come from pillow_check/
and grid_check/.
"""
import csv, math, sys
from decimal import ROUND_HALF_UP, Decimal
import numpy as np
import torch.nn as nn
import torchvision
from scipy import stats

PER_SEED, EARLIER, DATA = sys.argv[1], sys.argv[2], sys.argv[3]
CHANCE = 11.35          # always answering "1", the most common test digit
BATCH = 128
MCPHERSON = {'clean': 98.71, 'pixelate 2x2': 98.49, 'pixelate 4x4': 96.17, 'pixelate 7x7': 83.42,
             'pixelate 14x14': 52.13}   # [1], Table 2, paired by blocks per side
PRIVACY_BLUR = {'pixelate 7x7': 83.54, 'pixelate 14x14': 51.16, 'blur small': 97.58, 'blur large': 95.75}  # [2]

with open(PER_SEED) as fh:
    rows = list(csv.DictReader(fh))
CONDITIONS = list(dict.fromkeys(r['condition'] for r in rows))
N_TEST = int(rows[0]['n_test'])


def counts(model, condition, epoch, seeds=None):
    """Correct answers of one model per seed, in seed order."""
    pairs = sorted((int(r['seed']), int(r[f'correct_trained_on_{model}'])) for r in rows
                   if r['condition'] == condition and int(r['epoch']) == epoch)
    return np.array([c for s, c in pairs if seeds is None or s in seeds])


def acc(model, condition, epoch):
    return 100 * counts(model, condition, epoch) / N_TEST


def exact_mean(c):
    """Exact mean accuracy (%) of the integer counts c, rounded half up to two decimals."""
    return (Decimal(100 * int(c.sum())) / Decimal(len(c) * N_TEST)).quantize(Decimal('0.01'), ROUND_HALF_UP)


print('Table I and the README: test accuracy (%), mean ± sample SD over seeds [Table I, abstract]')
print(f'{"":15s}{"clean, 2 ep":>17s}{"obfusc., 2 ep":>17s}{"clean, 10 ep":>17s}{"obfusc., 10 ep":>17s}')
for c in CONDITIONS:
    cells = [f'{exact_mean(counts(m, c, e))} ± {acc(m, c, e).std(ddof=1):.2f}'
             for e in (2, 10) for m in ('clean', 'obfuscated')]
    print(f'{c:15s}' + ''.join(f'{x:>17s}' for x in cells))

print('\nObfuscation-trained model, accuracy on clean images minus accuracy under the condition, paired within seed'
      ' [abstract, IV-B, V]')
for c in ('blur sigma=4', 'pixelate 14x14'):
    for e in (2, 10):
        print(f'  {c}, epoch {e}: {exact_mean(counts("obfuscated", "clean", e) - counts("obfuscated", c, e))} points')

ratio = acc('obfuscated', 'pixelate 14x14', 2).mean() / CHANCE
print(f'\nPixelate 14x14, obfuscation-trained, epoch 2, as a multiple of chance ({CHANCE}%): {ratio:.2f} [IV-B]')

d = acc('clean', 'blur sigma=4', 2) - acc('clean', 'pixelate 7x7', 2)
print(f'\nClean-trained model, blur sigma=4 minus pixelate 7x7, epoch 2, per seed: {d.round(2).tolist()}; '
      f'lowest {d.min():.2f}, highest {d.max():+.2f} [V]')

v2, v10 = acc('clean', 'pixelate 7x7', 2), acc('clean', 'pixelate 7x7', 10)
half = stats.t.ppf(0.975, len(v2) - 1) * v2.std(ddof=1) * math.sqrt(1 + 1 / len(v2))
print(f'\nClean-trained model, pixelate 7x7: lowest of the {len(v2) + len(v10)} seed and epoch values '
      f'{min(v2.min(), v10.min()):.2f}; 95% prediction interval for a new seed after 2 epochs '
      f'[{v2.mean() - half:.2f}, {v2.mean() + half:.2f}] [V]')

print('\nSeeds 0 to 2 after 2 epochs minus the earlier three-seed run (' + EARLIER + ') [VI, README]')
with open(EARLIER) as fh:
    earlier = {r['condition']: r for r in csv.DictReader(fh)}
for m in ('clean', 'obfuscated'):
    diffs = {c: float(exact_mean(counts(m, c, 2, seeds={0, 1, 2}))) - float(earlier[c][f'trained_on_{m}_%'])
             for c in CONDITIONS}
    worst = max(diffs, key=lambda c: abs(diffs[c]))
    print(f'  trained on {m}: ' + ', '.join(f'{c} {x:+.2f}' for c, x in diffs.items())
          + f'; largest {abs(diffs[worst]):.2f} ({worst})')

train = torchvision.datasets.MNIST(DATA, train=True, download=True)
test = torchvision.datasets.MNIST(DATA, train=False, download=True)
n_train, n_test = np.bincount(train.targets.numpy(), minlength=10), np.bincount(test.targets.numpy(), minlength=10)
print(f'\nMNIST images per digit 0-9 [III-F, IV-C]\n  training: {n_train.tolist()} (total {n_train.sum()})'
      f'\n  test:     {n_test.tolist()} (total {n_test.sum()})')


class SmallCNN(nn.Module):   # the layers of the notebook's model (dl_obfuscation_v2.ipynb, cell 9), to count them
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)


updates = math.ceil(len(train) / BATCH)
print(f'\nModel parameters: {sum(p.numel() for p in SmallCNN().parameters()):,} [III-C]')
print(f'Parameter updates per epoch: {updates}; after 2 epochs: {2 * updates} [III-D]')
print(f'Half-width of a 95% t-interval for a mean of 10 seeds, in SDs: {stats.t.ppf(0.975, 9) / math.sqrt(10):.4f}'
      f' [III-F, Table I]')

print('\nComparisons with [1] and [2], obfuscation-trained model (my mean minus theirs) [abstract, V, VII]')
for e in (2, 10):
    for c in ('pixelate 2x2', 'pixelate 4x4', 'pixelate 7x7', 'pixelate 14x14', 'clean'):
        mine = float(exact_mean(counts('obfuscated', c, e)))
        line = f'  epoch {e:2d}, {c:15s} {mine:6.2f}: [1] {mine - MCPHERSON[c]:+.2f}'
        if c in PRIVACY_BLUR:
            line += f', [2] {mine - PRIVACY_BLUR[c]:+.2f}'
        print(line)
s2, s10 = (float(exact_mean(counts('obfuscated', 'blur sigma=4', e))) for e in (2, 10))
print(f'  blur sigma=4: {s2:.2f} after 2 epochs and {s10:.2f} after 10; [2] small radius {PRIVACY_BLUR["blur small"]}:'
      f' gap {PRIVACY_BLUR["blur small"] - s10:.2f} after 10 epochs, share of the 2-epoch gap closed '
      f'{(s10 - s2) / (PRIVACY_BLUR["blur small"] - s2):.2f}; [2] large radius {PRIVACY_BLUR["blur large"]}: '
      f'gap {PRIVACY_BLUR["blur large"] - s10:.2f}')
