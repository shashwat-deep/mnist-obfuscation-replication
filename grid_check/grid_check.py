"""Train the paper's model on digits pixelated to 4 blocks per side, on two grids, and test it after epochs 2 and 10.

    python grid_check/grid_check.py data grid_check/grid_check.jsonl

  pixelate 8x8 padded  the grid of McPherson et al.: a 2-pixel border makes the image 32x32, 8-pixel windows are
                       averaged, and the crop back to 28x28 leaves strips of 6, 8, 8 and 6 pixels. Their Table 1
                       shows MNIST as dark digits on white, so their "white border" is the background, 0 here.
  pixelate 7x7         the notebook's condition: four strips of 7 pixels.

The model, optimizer, batch size, seeds, training loop and test are copied from dl_obfuscation_v2.ipynb
(cells 1, 3, 7 and 9) and run on the CPU. Within a seed both grids start from the same initial weights and see
the training batches in the same order, so their difference is paired. Each output line is one condition,
seed and epoch, with the correct answers out of 10,000, hashes of the initial weights and batch order, and the
software and CPU. The script will not overwrite an existing output file.
"""
import datetime, hashlib, json, os, platform, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DATA, OUT = sys.argv[1], sys.argv[2]
if os.path.exists(OUT):
    sys.exit(f'{OUT} already exists: move or delete it before a new run')
THREADS = 4             # CPU results depend on the thread count, so it is fixed and recorded
torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
device = 'cpu'
EPOCHS = 10             # passes over the 60,000 training images
EVAL_EPOCHS = (2, 10)   # test the model after these epochs
BATCH = 128             # images per training step
LR = 1e-3               # Adam learning rate
SEEDS = [0, 1, 2, 3, 4]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def cpu_model():
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def pixelate(x, block):
    """Mosaic: average each block x block square, then scale it back up to 28 x 28."""
    return F.interpolate(F.avg_pool2d(x, block), scale_factor=block, mode='nearest')


CONDITIONS = {
    'pixelate 8x8 padded': lambda x: pixelate(F.pad(x, (2, 2, 2, 2)), 8)[:, :, 2:30, 2:30],
    'pixelate 7x7': lambda x: pixelate(x, 7),
}


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)    # 32 filters, 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)   # 64 filters, 3x3
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)                  # one output per digit

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)     # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)     # 14x14 -> 7x7
        x = self.drop(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


def sha(*tensors):
    """First 16 hex digits of the SHA-256 of the tensors' bytes."""
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def train_model(X, y, seed):
    """Train one model for EPOCHS epochs. After each epoch in EVAL_EPOCHS, pause and yield
    (epoch, model, hash of the initial weights, hash of the batch order so far). Testing the
    model during a pause draws no random numbers, so it does not change the rest of training."""
    set_seed(seed)
    model = SmallCNN().to(device)
    init_sha = sha(*model.state_dict().values())
    order = hashlib.sha256()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X), device=X.device)
        order.update(perm.cpu().numpy().tobytes())
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
        if epoch in EVAL_EPOCHS:
            yield epoch, model, init_sha, order.hexdigest()[:16]


@torch.no_grad()
def evaluate(model, X, y):
    """Number of correct answers, and the digit the model gave for every image if it gave only one (else None)."""
    model.eval()
    pred = torch.cat([model(X[i:i + 1000]).argmax(1) for i in range(0, len(X), 1000)])
    answer = int(pred[0]) if bool((pred == pred[0]).all()) else None
    return int((pred == y).sum()), answer


ENV = {
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'python': platform.python_version(),
    'platform': platform.platform(),
    'torch': torch.__version__,
    'torchvision': torchvision.__version__,
    'numpy': np.__version__,
    'device': device,
    'cpu': cpu_model(),
    'cpu_capability': torch.backends.cpu.get_cpu_capability(),
    'threads': torch.get_num_threads(),
}

train = torchvision.datasets.MNIST(DATA, train=True, download=True)
test = torchvision.datasets.MNIST(DATA, train=False, download=True)
Xtr = train.data.float().div(255).unsqueeze(1)   # 60000 x 1 x 28 x 28, pixel values 0..1
ytr = train.targets
Xte = test.data.float().div(255).unsqueeze(1)    # 10000 x 1 x 28 x 28
yte = test.targets
print(json.dumps(ENV), flush=True)
with open(OUT, 'x') as out:
    for name, f in CONDITIONS.items():
        Xtr_o, Xte_o = f(Xtr), f(Xte)
        for s in SEEDS:
            t0 = time.time()
            for epoch, model, init_sha, order_sha in train_model(Xtr_o, ytr, s):
                correct, _ = evaluate(model, Xte_o, yte)
                rec = {'condition': name, 'seed': s, 'epoch': epoch, 'n_test': len(yte), 'correct': correct,
                       'init_sha': init_sha, 'order_sha': order_sha, 'seconds': round(time.time() - t0), 'env': ENV}
                out.write(json.dumps(rec) + '\n')
                out.flush()
                print(f'{name:20s} seed {s} epoch {epoch:2d}: {100 * correct / len(yte):.2f}% ({rec["seconds"]} s)',
                      flush=True)
print('done', flush=True)
