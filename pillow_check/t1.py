import importlib.util, sys, numpy as np, torchvision
spec = importlib.util.spec_from_file_location('emul', sys.argv[1] + '/emul.py'); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
rng = np.random.default_rng(0)
test = torchvision.datasets.MNIST(sys.argv[2], train=False, download=False).data.numpy()
# 1. bit-exactness on random images and MNIST
for r in [0.5, 1, 2, 2.8 * 2 ** 0.5, 4, 7.3]:
    bad = 0
    for _ in range(30):
        h, w = rng.integers(5, 40, 2)
        im = rng.integers(0, 256, (h, w), dtype=np.uint8)
        bad += int((E.pil(im, r) != E.emul(im, r)).sum())
    sub = test[:500]
    bad_m = int((np.stack([E.pil(a, r) for a in sub]) != E.emul(sub, r)).sum())
    fr = E.gb_radius(r); rr, ww, fw = E.weights(fr)
    k, _ = E.kernel1d(r)
    t = np.arange(len(k)) - len(k) // 2
    print(f"r={r:.4f} box radius={float(fr):.5f} int={rr} ww={ww} fw={fw} sum1={(2*rr+1)*ww+2*fw} | 1-pass taps={2*rr+3} 3-pass taps={len(k)} "
          f"kernel SD={np.sqrt((k*t**2).sum()/k.sum()):.4f} sum={k.sum():.8f} | mismatched px rand={bad} mnist={bad_m}")
