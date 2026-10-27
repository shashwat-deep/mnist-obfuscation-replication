"""10-epoch obfuscation-trained attacker on three sigma~4 blurs, paired by seed, on the CPU.

  notebook  the notebook's blur(x, 4): float32, reflect padding, 25 taps
  pillow    PIL ImageFilter.GaussianBlur(radius = 28*sqrt(2)/10), as in Privacy Blur: uint8 in and out
  zeropad   the notebook's blur with zero padding instead of reflect (isolates the edge effect)

Model, optimiser, batch size, seeds and training loop are the notebook's; accuracy is counted in integers.
Will not overwrite an existing output file.
"""
import datetime, json, os, sys, time, random, platform
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
import PIL
from PIL import Image, ImageFilter

torch.set_num_threads(4)
DATA, OUT = sys.argv[1], sys.argv[2]
if os.path.exists(OUT):
    sys.exit(f'{OUT} already exists: move or delete it before a new run')
EPOCHS, BATCH, LR, SEEDS = 10, 128, 1e-3, [0, 1, 2]

def cpu_model():
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()

ENV = dict(torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
           pillow=PIL.__version__, python=platform.python_version(), device='cpu', threads=torch.get_num_threads(),
           cpu=cpu_model(), cpu_capability=torch.backends.cpu.get_cpu_capability(), platform=platform.platform(),
           started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

train = torchvision.datasets.MNIST(DATA, train=True, download=True)
test = torchvision.datasets.MNIST(DATA, train=False, download=True)

def blur(x, sigma, mode='reflect'):
    radius = int(3 * sigma)
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum(); k = len(g)
    pad = (radius, radius, radius, radius)
    x = F.pad(x, pad, mode='reflect') if mode == 'reflect' else F.pad(x, pad)
    x = F.conv2d(x, g.view(1, 1, 1, k)); x = F.conv2d(x, g.view(1, 1, k, 1))
    return x

def pil_blur(u8):  # u8: N x 28 x 28 uint8 tensor
    r = 28 * 2 ** 0.5 / 10
    out = np.stack([np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(r))) for a in u8.numpy()])
    return torch.from_numpy(out).float().div(255).unsqueeze(1)

ARMS = {
    'notebook': lambda u8: blur(u8.float().div(255).unsqueeze(1), 4),
    'pillow': pil_blur,
    'zeropad': lambda u8: blur(u8.float().div(255).unsqueeze(1), 4, mode='zeros'),
}

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1); self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128); self.drop = nn.Dropout(0.25); self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2); x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop(F.relu(self.fc1(x.flatten(1)))); return self.fc2(x)

def train_model(X, y, seed):
    set_seed(seed)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X), device=X.device)
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); loss = F.cross_entropy(model(X[idx]), y[idx]); loss.backward(); opt.step()
    return model

@torch.no_grad()
def correct(model, X, y):
    model.eval()
    return sum((model(X[i:i + 1000]).argmax(1) == y[i:i + 1000]).sum().item() for i in range(0, len(X), 1000))

with open(OUT, 'x') as out:
    for arm, f in ARMS.items():
        Xtr, Xte = f(train.data), f(test.data)
        for s in SEEDS:
            t0 = time.time()
            model = train_model(Xtr, train.targets, s)
            c = correct(model, Xte, test.targets)
            rec = dict(arm=arm, seed=s, epochs=EPOCHS, correct=c, n=len(Xte), acc=100 * c / len(Xte),
                       secs=round(time.time() - t0, 1), env=ENV)
            out.write(json.dumps(rec) + '\n'); out.flush(); print(rec, flush=True)
print('done', flush=True)
