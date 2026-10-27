import importlib.util, sys, numpy as np, torch, torch.nn.functional as F, torchvision
spec = importlib.util.spec_from_file_location('emul', sys.argv[1] + '/emul.py'); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
X8 = torchvision.datasets.MNIST(sys.argv[2], train=False, download=False).data.numpy()
Xf = X8 / 255.0

def nb_blur(x, sigma, mode='reflect'):
    radius = int(3 * sigma); t = torch.arange(-radius, radius + 1, dtype=torch.float64)
    g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum(); k = len(g)
    x = F.pad(torch.as_tensor(x).unsqueeze(1), (radius,) * 4, mode=mode)
    return F.conv2d(F.conv2d(x, g.view(1, 1, 1, k)), g.view(1, 1, k, 1))[:, 0].numpy()

def conv_once(x, k, mode):
    R = len(k) // 2; xt = F.pad(torch.as_tensor(x).unsqueeze(1), (R,) * 4, mode=mode); kk = torch.as_tensor(k)
    return F.conv2d(F.conv2d(xt, kk.view(1, 1, 1, -1)), kk.view(1, 1, -1, 1))[:, 0].numpy()

def stat(a, b):  # in gray levels (0..255)
    d = np.abs(a - b)
    return f"mean|d|={d.mean():.3f}  p99={np.percentile(d, 99):.2f}  max={d.max():.2f}  frac px differ(>0.5)={(d > 0.5).mean():.3f}"

for r in [4, 2.8 * 2 ** 0.5]:
    k, _ = E.kernel1d(r)
    A  = nb_blur(Xf, 4) * 255                       # notebook float Gaussian, reflect
    B  = np.round(A)                                # the "8-bit ablation": round once at end
    C  = conv_once(Xf, k, 'reflect') * 255          # PIL's box-composite kernel, reflect, float
    Cr = conv_once(Xf, k, 'replicate') * 255        # PIL kernel, replicate ONCE (= zero pad for MNIST)
    D  = E.emul(X8, r, rnd=False)                   # PIL pipeline in float (replicate each pass)
    D8 = np.floor(D + 0.5)                          # PIL pipeline, rounded once at end
    P  = np.stack([E.pil(a, r) for a in X8]).astype(np.float64)
    print(f"== PIL radius {r:.4f} vs notebook sigma 4 (all in gray levels) ==")
    print("  total: ablation round(notebook) vs PIL      ", stat(B, P))
    print("  kernel shape: Gaussian vs box-composite (both reflect, float)", stat(A, C))
    print("  edges: reflect-once vs replicate-each-pass (box, float)     ", stat(C, D))
    print("  edges: replicate-once vs replicate-each-pass (box, float)   ", stat(Cr, D))
    print("  rounding: per-pass (PIL) vs once-at-end (same pipeline)     ", stat(P, D8))
    print("  rounding: PIL vs float pipeline                             ", stat(P, D))
    print("  rounding: once-at-end vs float                              ", stat(D8, D))
    print("  rounding bias: mean(P - D) =", f"{(P - D).mean():.4f}", " mean(D8 - D) =", f"{(D8 - D).mean():.4f}")
    # interior only (exclude 6-px border) to isolate non-edge differences
    sl = (slice(None), slice(6, 22), slice(6, 22))
    print("  interior 16x16: ablation vs PIL", stat(B[sl], P[sl]), "| per-pass vs end rounding", stat(P[sl], D8[sl]))
