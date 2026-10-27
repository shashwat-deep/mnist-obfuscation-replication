import importlib.util, sys, numpy as np, torch, torch.nn.functional as F, torchvision
spec = importlib.util.spec_from_file_location('emul', sys.argv[1] + '/emul.py'); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
test = torchvision.datasets.MNIST(sys.argv[2], train=False, download=False).data.numpy()

def nb_blur(x, sigma, mode='reflect'):  # notebook's blur, float64, x: N x H x W float in [0,1]
    radius = int(3 * sigma)
    t = torch.arange(-radius, radius + 1, dtype=torch.float64)
    g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum(); k = len(g)
    x = torch.as_tensor(x, dtype=torch.float64).unsqueeze(1)
    pad = (radius,) * 4
    x = F.pad(x, pad, mode=mode) if mode != 'zeros' else F.pad(x, pad)
    x = F.conv2d(x, g.view(1, 1, 1, k)); x = F.conv2d(x, g.view(1, 1, k, 1))
    return x[:, 0].numpy()

def conv_once(x, k, mode):  # composite separable kernel applied once, given padding
    R = len(k) // 2
    xt = torch.as_tensor(x, dtype=torch.float64).unsqueeze(1)
    kk = torch.as_tensor(k)
    m = {'zeros': 'constant', 'edge': 'replicate', 'reflect': 'reflect'}[mode]
    xt = F.pad(xt, (R,) * 4, mode=m)
    xt = F.conv2d(xt, kk.view(1, 1, 1, -1)); xt = F.conv2d(xt, kk.view(1, 1, -1, 1))
    return xt[:, 0].numpy()

def emul_float_zero(img, radius):  # 3-pass extended box, float, ZERO padding each pass
    fr = E.gb_radius(radius); r, ww, fw = E.weights(fr)
    x = img.astype(np.float64)
    for ax in (-1, -2):
        for _ in range(3):
            x = np.swapaxes(x, ax, -1)
            n = x.shape[-1]; p = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(r + 1, r + 1)])
            acc = sum(p[..., r + 1 + d: r + 1 + d + n] for d in range(-r, r + 1))
            x = np.swapaxes((acc * ww + (p[..., :n] + p[..., 2 * r + 2:2 * r + 2 + n]) * fw) / 2 ** 24, ax, -1)
    return x

print("== empirical SD of PIL impulse response (255 impulse, 101x101) ==")
for r in [1, 2, 4]:
    im = np.zeros((101, 101), np.uint8); im[50, 50] = 255
    o = E.pil(im, r).astype(float); m = o.sum(0); t = np.arange(101) - 50
    k, _ = E.kernel1d(r)
    print(f"r={r}: PIL 8-bit SD={np.sqrt((m*t**2).sum()/m.sum()):.4f}  float pipeline SD={np.sqrt(((k*(np.arange(len(k))-len(k)//2)**2).sum())):.4f}  "
          f"PIL ink kept={o.sum()/255:.4f}  nonzero px={int((o>0).sum())}")

print("\n== bright line at column 1 (28x28, full height, value 255) ==")
for r in [4, 2.8 * 2 ** 0.5]:
    im = np.zeros((28, 28), np.uint8); im[:, 1] = 255
    k, _ = E.kernel1d(r)
    rows = {
        'PIL actual (uint8, round each pass, replicate each pass)': E.pil(im, r) / 255,
        'emul float (no rounding), replicate each pass':          E.emul(im, r, rnd=False) / 255,
        'emul float, ZERO pad each pass':                         emul_float_zero(im[None], r)[0] / 255,
        'PIL composite kernel once, replicate(=zero here)':       conv_once(im[None] / 255, k, 'edge')[0],
        'PIL composite kernel once, reflect':                     conv_once(im[None] / 255, k, 'reflect')[0],
        'notebook Gaussian 25 taps, zero':                        nb_blur(im[None] / 255, 4, 'zeros')[0],
        'notebook Gaussian 25 taps, replicate':                   nb_blur(im[None] / 255, 4, 'replicate')[0],
        'notebook Gaussian 25 taps, reflect':                     nb_blur(im[None] / 255, 4, 'reflect')[0],
    }
    print(f"-- r={r:.4f}")
    for kname, o in rows.items():
        print(f"   {kname:58s} mass kept = {o.sum() / 28:.4f}")
    # per pass mass trace (float, replicate)
    fr = E.gb_radius(r); x = im[None].astype(np.float64)
    trace = []
    for _ in range(3):
        x = E.hpass(x, fr, rnd=False); trace.append(x.sum() / (255 * 28))
    xi = im[None].copy(); trace_i = []
    for _ in range(3):
        xi = E.hpass(xi, fr, rnd=True); trace_i.append(xi.astype(float).sum() / (255 * 28))
    print(f"   per-pass mass, float replicate: {np.round(trace, 4)}; uint8: {np.round(trace_i, 4)}; col0 after pass1 = {x[0,0,0]:.1f}")

print("\n== MNIST test set (10000 images): total ink out / ink in ==")
X8 = test; Xf = X8 / 255.0; ink = Xf.sum()
for r in [4, 2.8 * 2 ** 0.5]:
    k, _ = E.kernel1d(r)
    P = np.stack([E.pil(a, r) for a in X8]) / 255
    Ef = E.emul(X8, r, rnd=False) / 255
    print(f"-- r={r:.4f}")
    print(f"   PIL actual                                  {P.sum()/ink:.4f}")
    print(f"   same pipeline in float (no rounding)        {Ef.sum()/ink:.4f}")
    print(f"   3-pass box, float, ZERO pad each pass       {emul_float_zero(X8, r).sum()/255/ink:.4f}")
    print(f"   composite kernel once, zero/replicate pad   {conv_once(Xf, k, 'zeros').sum()/ink:.4f}")
    print(f"   composite kernel once, reflect pad          {conv_once(Xf, k, 'reflect').sum()/ink:.4f}")
print(f"   notebook Gaussian s=4, reflect              {nb_blur(Xf, 4).sum()/ink:.4f}")
print(f"   notebook Gaussian s=4, zero                 {nb_blur(Xf, 4, 'zeros').sum()/ink:.4f}")
print(f"   notebook Gaussian s=4, replicate            {nb_blur(Xf, 4, 'replicate').sum()/ink:.4f}")
print(f"   border ring (outer 1px) of raw MNIST test: mean={X8[:, [0, -1], :].mean():.3f} / {X8[:, :, [0, -1]].mean():.3f}; any nonzero frac={(np.concatenate([X8[:,[0,-1],:].reshape(len(X8),-1), X8[:,:,[0,-1]].reshape(len(X8),-1)],1)>0).any(1).mean():.4f}")
