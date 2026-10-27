# McPherson grid check

McPherson et al. made their MNIST images 32×32 by adding a 2-pixel border before mosaicing, so their 8-pixel
windows cut a 28×28 digit into strips of 6, 8, 8 and 6 pixels, where the notebook's 7×7 blocks cut it into four
strips of 7. This check trains the paper's model on both grids on a CPU, with seeds 0 to 4, and tests it after
epochs 2 and 10. Shashwat Deep ran it on a laptop CPU (Intel Core i7-14650HX, 4 threads) with PyTorch 2.14.1.

| File | What it is |
|---|---|
| `grid_check.py` | Trains the model of `dl_obfuscation_v2.ipynb` on obfuscated images under each grid: `pixelate 8x8 padded` (the grid of McPherson et al., cropped back to 28×28) and `pixelate 7x7` (the notebook's). The model, optimizer, batch size, seeds, training loop and test are copied from the notebook. Each output line is one grid, seed and epoch, with the correct answers out of 10,000, hashes of the initial weights and batch order, and the software and CPU. |
| `grid_check.jsonl` | Its 20 results. |
| `grid_summary.py`, `grid_summary.txt` | Means, sample SDs and 95% t-intervals for each grid, the difference between the grids paired by seed, and the padded grid against the 83.42% of McPherson et al. at 8×8 windows, and what it printed. Given `results_per_seed.csv`, it also checks that each seed started from the same initial weights as in the GPU run, which they all did. |

Test accuracy of the model trained on obfuscated images (%), mean over seeds 0 to 4 [95% interval]:

| | After 2 epochs | After 10 epochs |
|---|---|---|
| Grid of McPherson et al. (8×8 windows, padded) | 79.71 [79.36, 80.06] | 83.58 [83.37, 83.79] |
| Paper's grid (7×7 blocks) | 83.88 [83.08, 84.68] | 87.39 [86.93, 87.85] |
| Difference between the grids, paired by seed | −4.17 [−5.17, −3.17] | −3.81 [−4.38, −3.24] |
| Grid of McPherson et al. minus their 83.42% | −3.71 [−4.06, −3.36] | +0.16 [−0.05, +0.37] |

## Run

You need the packages in `requirements-cpu-checks.txt` (see its first lines). Run from the main folder; `data` is
the MNIST folder that torchvision uses, and `grid_check.py` downloads MNIST into it if it is missing:

```bash
.venv/bin/python grid_check/grid_check.py data grid_check/grid_check.jsonl
.venv/bin/python grid_check/grid_summary.py grid_check/grid_check.jsonl results_per_seed.csv
```

`grid_check.py` trains ten models on 4 CPU threads and will not overwrite an existing output file. CPU results
depend on the thread count and the CPU, so the output records both. On the laptop it took about 14 minutes.
