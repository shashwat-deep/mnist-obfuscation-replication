"""Redraw Figure 1 (one test image under each condition) for one IEEE column.

Usage: python make_fig_examples.py <t10k-images-idx3-ubyte.gz> <fig_examples.png>
(After the notebook runs in Colab, the file is at data/MNIST/raw/t10k-images-idx3-ubyte.gz.)

The obfuscations are line-for-line numpy ports of the notebook's functions:
pixelate = F.interpolate(F.avg_pool2d(x, b), scale_factor=b, mode='nearest')
blur     = separable Gaussian, radius int(3*sigma), reflect padding
mask     = torch.zeros_like(x)
"""
import gzip, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
with gzip.open(src, "rb") as f:
    raw = f.read()
magic, n, rows, cols = np.frombuffer(raw[:16], dtype=">i4")
assert magic == 2051 and (n, rows, cols) == (10000, 28, 28)
X = np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, 28, 28).astype(np.float32) / 255.0
x = X[0]                                    # Xte[:1] in the notebook

def pixelate(x, b):
    m = x.reshape(28 // b, b, 28 // b, b).mean(axis=(1, 3))   # avg_pool2d(x, b)
    return np.repeat(np.repeat(m, b, axis=0), b, axis=1)     # nearest upsample

def blur(x, sigma):
    r = int(3 * sigma)
    t = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    xp = np.pad(x, r, mode="reflect")                        # F.pad(..., 'reflect')
    h = np.apply_along_axis(lambda v: np.convolve(v, g, "valid"), 1, xp)  # rows
    return np.apply_along_axis(lambda v: np.convolve(v, g, "valid"), 0, h)  # cols

panels = [  # (label, image); two rows of five, both rows start from the clean image
    ("Clean", x), ("Pixelate\n2×2", pixelate(x, 2)), ("Pixelate\n4×4", pixelate(x, 4)),
    ("Pixelate\n7×7", pixelate(x, 7)), ("Pixelate\n14×14", pixelate(x, 14)),
    ("Clean", x), ("Blur\nσ = 1", blur(x, 1)), ("Blur\nσ = 2", blur(x, 2)),
    ("Blur\nσ = 4", blur(x, 4)), ("Solid\nmask", np.zeros_like(x)),
]
for lab, im in panels:
    assert im.shape == (28, 28) and im.min() >= -1e-6 and im.max() <= 1 + 1e-6, lab
print("distinct values: px14 =", len(np.unique(np.round(pixelate(x, 14), 6))),
      "px7 =", len(np.unique(np.round(pixelate(x, 7), 6))))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "mathtext.fontset": "stix"})
fig, axes = plt.subplots(2, 5, figsize=(3.35, 1.95))
for ax, (lab, im) in zip(axes.flat, panels):
    ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(lab, fontsize=8, linespacing=1.0, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
plt.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01, wspace=0.08, hspace=0.42)
fig.savefig(out, dpi=600)
print("saved", out)
