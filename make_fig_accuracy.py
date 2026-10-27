"""Draws Fig. 2 of the paper from the per-seed results.

    python make_fig_accuracy.py results_per_seed.csv fig_accuracy.png

Bars: mean test accuracy over seeds after two epochs, with 95% t-intervals.
Circles: mean after ten epochs. Needs NumPy, SciPy and Matplotlib.
"""
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

src, out = sys.argv[1], sys.argv[2]
CONDITIONS = [('clean', 'Clean'), ('pixelate 2x2', 'Pixelate\n2×2'), ('pixelate 4x4', 'Pixelate\n4×4'),
              ('pixelate 7x7', 'Pixelate\n7×7'), ('pixelate 14x14', 'Pixelate\n14×14'), ('blur sigma=1', 'Blur\nσ = 1'),
              ('blur sigma=2', 'Blur\nσ = 2'), ('blur sigma=4', 'Blur\nσ = 4'), ('solid mask', 'Solid\nmask')]
MODELS = [('clean', '0.82', 'Trained on clean images'), ('obfuscated', '0.35', 'Trained on obfuscated images')]
CHANCE = 11.35   # always answering "1", the most common test digit (1,135 of 10,000 test images)
with open(src) as fh:
    rows = list(csv.DictReader(fh))


def accuracies(condition, epoch, model):
    return np.array([100 * int(r[f'correct_trained_on_{model}']) / int(r['n_test'])
                     for r in rows if r['condition'] == condition and int(r['epoch']) == epoch])


def half_width(x):
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                     'font.size': 8, 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(figsize=(6.9, 1.9), layout='constrained')
x = np.arange(len(CONDITIONS))
w = 0.38
for (model, shade, _), dx in zip(MODELS, (-w / 2, w / 2)):
    two = [accuracies(c, 2, model) for c, _ in CONDITIONS]
    ax.bar(x + dx, [a.mean() for a in two], w, yerr=[half_width(a) for a in two], color=shade, edgecolor='black',
           linewidth=0.5, error_kw={'elinewidth': 0.7, 'capsize': 1.5, 'capthick': 0.7})
    ax.plot(x + dx, [accuracies(c, 10, model).mean() for c, _ in CONDITIONS], 'o', ms=3.2,
            mfc='white', mec='black', mew=0.6, zorder=3)
ax.axhline(CHANCE, color='black', linestyle='--', linewidth=0.7, zorder=0)
ax.set_xticks(x, [label for _, label in CONDITIONS])
ax.tick_params(axis='x', length=0)
ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
ax.set_ylim(0, 105)
ax.set_yticks(range(0, 101, 20))
ax.set_ylabel('Test accuracy (%)')
ax.spines[['top', 'right']].set_visible(False)
handles = [Patch(facecolor=shade, edgecolor='black', linewidth=0.5, label=label) for _, shade, label in MODELS]
handles += [Line2D([], [], marker='o', ls='none', ms=3.2, mfc='white', mec='black', mew=0.6, label='After ten epochs'),
            Line2D([], [], color='black', ls='--', lw=0.7, label=f'Chance ({CHANCE}%)')]
fig.legend(handles=handles, loc='outside upper center', ncol=4, frameon=False, handlelength=1.6, columnspacing=1.8)
fig.savefig(out, dpi=600)
