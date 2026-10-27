"""Measure Pillow's GaussianBlur against the notebook's blur on 8-bit images.

Usage: python -I pil_measure.py <MNIST data dir, as passed to torchvision.datasets.MNIST>

1. Radius semantics: the SD of Pillow's effective kernel, recovered from its step response.
2. Edge handling: how much of a bright line next to the border stays inside the image,
   for Pillow and for a true Gaussian with zero, reflect and replicate padding.
3. MNIST: total ink after blurring the 10,000 test images, and how far Pillow's output
   (radius = diagonal/10, as in Privacy Blur) is from the notebook's blur and its variants.
"""
import sys
import numpy as np
import torch, torch.nn.functional as F, torchvision
from PIL import Image, ImageFilter

DATA = sys.argv[1]


def pil_blur(a, r):
    """Pillow's GaussianBlur on an 8-bit image, returned as float in 0..255."""
    return np.asarray(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), float)


def tblur(x, s, mode):
    """The notebook's separable Gaussian (radius int(3*sigma)) with a choice of padding."""
    r = int(3 * s)
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-t ** 2 / (2 * s ** 2)); g /= g.sum(); k = len(g)
    x = F.pad(x, (r, r, r, r), mode=mode) if mode != 'zeros' else F.pad(x, (r, r, r, r))
    return F.conv2d(F.conv2d(x, g.view(1, 1, 1, k)), g.view(1, 1, k, 1))


# 1. Radius semantics via the step response: one row = 255 * CDF of the 1-D kernel.
step = np.zeros((9, 201)); step[:, 100:] = 255
for r in (1, 2, 4, 28 * 2 ** 0.5 / 10):
    c = pil_blur(step, r)[4] / 255
    p = np.diff(c); x = np.arange(len(p)); mu = (x * p).sum() / p.sum()
    sd = np.sqrt(((x - mu) ** 2 * p).sum() / p.sum())
    print(f"radius={r:.3f}: SD of Pillow's effective kernel = {sd:.3f}, support = {np.count_nonzero(p)} taps")

# 2. Edge handling: bright vertical line at column 1; share of its mass left in row 20.
line = np.zeros((40, 40)); line[:, 1] = 255
lt = torch.tensor(line, dtype=torch.float32)[None, None]
kept = {m: tblur(lt, 4, m)[0, 0, 20].sum().item() / 255 for m in ('zeros', 'reflect', 'replicate')}
print(f"edge line, Pillow r=4 keeps {pil_blur(line, 4)[20].sum() / 255:.3f} of its mass | "
      f"Gaussian with zero {kept['zeros']:.3f}, reflect {kept['reflect']:.3f}, replicate {kept['replicate']:.3f}")

# 3. MNIST test set: Pillow (as Privacy Blur: radius = diagonal/10, uint8 output) vs the notebook's blur.
X = torchvision.datasets.MNIST(DATA, train=False).data.numpy()
pil = np.stack([pil_blur(x, 28 * 2 ** 0.5 / 10) for x in X]) / 255
Xt = torch.from_numpy(X).float().div(255)[:, None]
nb, zp, rp = (tblur(Xt, 4, m)[:, 0].numpy() for m in ('reflect', 'zeros', 'replicate'))
o = Xt.sum().item()
print(f"MNIST ink vs original: Pillow {pil.sum() / o:.4f} | notebook (reflect) {nb.sum() / o:.4f} | "
      f"zero pad {zp.sum() / o:.4f} | replicate {rp.sum() / o:.4f}")
for name, ref in (('notebook reflect', nb), ('zero pad', zp), ('replicate', rp)):
    d = np.abs(pil - ref)
    print(f"  |Pillow - {name:16s}|: mean {d.mean():.4f}, border (4 px) mean {d[:, np.r_[0:4, 24:28]].mean():.4f}, "
          f"max {d.max():.3f}")
