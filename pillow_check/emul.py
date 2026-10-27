"""Bit-exact NumPy emulation of Pillow 12.3.0 src/libImaging/BoxBlur.c GaussianBlur for mode 'L'."""
import math, numpy as np
from PIL import Image, ImageFilter
f32 = np.float32

def gb_radius(radius, passes=3):
    radius = f32(radius)
    sigma2 = f32(radius * radius / f32(passes))
    L = f32(math.sqrt(12.0 * float(sigma2) + 1.0))
    l = f32(math.floor((float(L) - 1.0) / 2.0))
    a = f32((f32(2) * l + f32(1)) * (l * (l + f32(1)) - f32(3) * sigma2))
    a = f32(a / (f32(6) * (sigma2 - (l + f32(1)) * (l + f32(1)))))
    return f32(l + a)

def weights(fr):
    r = int(fr)
    ww = int(f32(f32(1 << 24) / (fr * f32(2) + f32(1))))
    fw = ((1 << 24) - (r * 2 + 1) * ww) // 2
    return r, ww, fw

def hpass(img, fr, rnd=True):
    """one horizontal extended-box pass along last axis, replicate edges, like ImagingLineBoxBlur8"""
    r, ww, fw = weights(fr)
    p = np.pad(img, [(0, 0)] * (img.ndim - 1) + [(r + 1, r + 1)], mode='edge')
    n = img.shape[-1]
    if rnd:
        p = p.astype(np.int64)
        acc = sum(p[..., r + 1 + d: r + 1 + d + n] for d in range(-r, r + 1))
        bulk = acc * ww + (p[..., 0:n] + p[..., 2 * r + 2:2 * r + 2 + n]) * fw
        return ((bulk + (1 << 23)) >> 24).astype(np.uint8)
    p = p.astype(np.float64)
    acc = sum(p[..., r + 1 + d: r + 1 + d + n] for d in range(-r, r + 1))
    return (acc * ww + (p[..., 0:n] + p[..., 2 * r + 2:2 * r + 2 + n]) * fw) / 2 ** 24

def emul(img, radius, passes=3, rnd=True, edge='edge'):
    fr = gb_radius(radius, passes)
    x = img if rnd else img.astype(np.float64)
    for _ in range(passes):
        x = hpass(x, fr, rnd)
    x = np.swapaxes(x, -1, -2)
    for _ in range(passes):
        x = hpass(x, fr, rnd)
    return np.swapaxes(x, -1, -2)

def pil(img, radius):
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)))

def kernel1d(radius, passes=3):
    """composite 1D kernel (float) of the passes-fold extended box"""
    fr = gb_radius(radius, passes); r, ww, fw = weights(fr)
    k1 = np.array([fw] + [ww] * (2 * r + 1) + [fw], dtype=np.float64) / 2 ** 24
    k = np.array([1.0])
    for _ in range(passes):
        k = np.convolve(k, k1)
    return k, (fr, r, ww, fw, k1)
