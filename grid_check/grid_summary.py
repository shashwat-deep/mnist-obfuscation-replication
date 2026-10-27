"""Summarise grid_check.jsonl: accuracy on each grid after epochs 2 and 10, the paired difference between the grids,
and the padded grid against the 83.42% of McPherson et al. at 8x8 windows.

    python grid_check/grid_summary.py grid_check/grid_check.jsonl results_per_seed.csv

The second argument is optional. With it, the script checks that every seed started from the same initial weights
as in the GPU run, and prints the GPU run's 7x7 results for the same seeds.
"""
import csv, json, sys
from decimal import ROUND_HALF_UP, Decimal
import numpy as np
from scipy import stats

MCPHERSON_8X8 = 83.42   # [1], Table 2: MNIST mosaiced with 8x8 windows on 32x32 images
PADDED, PAPER = 'pixelate 8x8 padded', 'pixelate 7x7'

R = [json.loads(line) for line in open(sys.argv[1])]
correct, init = {}, {}   # (condition, epoch) -> {seed: correct answers}; seed -> {initial-weight hashes}
for r in R:
    correct.setdefault((r['condition'], r['epoch']), {})[r['seed']] = r['correct']
    init.setdefault(r['seed'], set()).add(r['init_sha'])
N_TEST = R[0]['n_test']
print(f'{len(R)} lines; torch {R[0]["env"]["torch"]}, {R[0]["env"]["threads"]} threads, {R[0]["env"]["cpu"]}')


def describe(x, sign=''):
    """Mean, sample SD and 95% t-interval of x; sign='+' prints the sign of the mean and the bounds."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return f'{x.mean():{sign}.2f} (n = 1)'
    sd = x.std(ddof=1)
    half = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))
    return f'{x.mean():{sign}.2f} ± {sd:.2f} [{x.mean() - half:{sign}.2f}, {x.mean() + half:{sign}.2f}]'


def exact_mean(counts, n_test=N_TEST):
    """Mean accuracy (%) of integer counts, exact, rounded half up to two decimals."""
    return (Decimal(100 * sum(counts)) / Decimal(len(counts) * n_test)).quantize(Decimal('0.01'), ROUND_HALF_UP)


for epoch in sorted({e for _, e in correct}):
    seeds = sorted(set(correct.get((PADDED, epoch), {})) & set(correct.get((PAPER, epoch), {})))
    print(f'\nAfter {epoch} epochs, seeds {seeds}; accuracy (%) per seed, then mean ± SD [95% interval]')
    acc = {}
    for c in (PADDED, PAPER):
        acc[c] = np.array([100 * correct[c, epoch][s] / N_TEST for s in seeds])
        print(f'  {c:20s} {acc[c].round(2).tolist()}  mean {exact_mean([correct[c, epoch][s] for s in seeds])}  '
              f'{describe(acc[c])}')
    print(f'  {PADDED} minus {PAPER}, paired: {(acc[PADDED] - acc[PAPER]).round(2).tolist()}  '
          f'{describe(acc[PADDED] - acc[PAPER], "+")}')
    print(f'  {PADDED} minus {MCPHERSON_8X8} of [1]: {describe(acc[PADDED] - MCPHERSON_8X8, "+")}')

shared = all(len(h) == 1 for h in init.values())
print('\nWithin each seed, both grids started from the same initial weights:', shared)
if len(sys.argv) > 2:
    with open(sys.argv[2]) as fh:
        gpu = [r for r in csv.DictReader(fh) if r['condition'] == PAPER]
    same = all(init[int(r['seed'])] == {r['init_sha']} for r in gpu if int(r['seed']) in init)
    print(f'Same initial weights as the GPU run ({sys.argv[2]}) for these seeds:', same)
    n_gpu = int(gpu[0]['n_test'])
    for epoch in sorted({int(r['epoch']) for r in gpu}):
        rows = sorted((int(r['seed']), int(r['correct_trained_on_obfuscated'])) for r in gpu
                      if int(r['epoch']) == epoch and int(r['seed']) in init)
        print(f'  GPU run, {PAPER}, epoch {epoch}, seeds {[s for s, _ in rows]}: '
              f'{[round(100 * c / n_gpu, 2) for _, c in rows]}  mean {exact_mean([c for _, c in rows], n_gpu)}')
