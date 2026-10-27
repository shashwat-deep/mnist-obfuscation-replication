# MNIST obfuscation replication

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23692156.svg)](https://doi.org/10.5281/zenodo.23692156)

Code and results for the paper [*Can a Small Convolutional Network Still Read Blurred and
Pixelated Digits? A Replication of McPherson et al. (2016)*](paper.pdf) by Shashwat Deep.

The experiment trains a small CNN on MNIST digits that are pixelated (blocks of 2, 4, 7 and
14 pixels), blurred with a Gaussian of σ = 1, 2 or 4, or covered by a solid mask. It compares
a model trained on clean images with one trained on obfuscated images. The second is the
attack of McPherson, Shokri and Shmatikov, *Defeating Image Obfuscation with Deep Learning*
(arXiv:1609.00408, 2016).

## Files

| File | What it is |
|---|---|
| `paper.pdf` | The paper (5 pages, IEEE conference format). |
| `dl_obfuscation_v2.ipynb` | The experiment in the paper: 9 conditions, seeds 0 to 9 on a GPU, every model trained for 10 epochs and tested after epochs 2 and 10, with deterministic GPU kernels. Writes `results_per_seed.csv` and `run_info.json`. This copy keeps the printed outputs of the run reported in the paper (Google Colab, NVIDIA T4, 2026-10-22). |
| `results_per_seed.csv` | Results of that run, one row per condition, seed and epoch: correct answers out of 10,000 for both models, the digit a model gave for every test image when it gave only one, and hashes of the initial weights and batch order. |
| `run_info.json` | The run's software versions, GPU and CPU, settings and training time, the SHA-256 of `results_per_seed.csv`, and hashes of the obfuscated images. |
| `run_info_second_run.json` | The same record for a second run on a T4, which reproduced `results_per_seed.csv` byte for byte. |
| `fig_accuracy.png`, `make_fig_accuracy.py` | Fig. 2 of the paper, drawn from `results_per_seed.csv`. |
| `fig_examples.png`, `make_fig_examples.py` | Fig. 1 of the paper, redrawn for one journal column with NumPy ports of the notebook's obfuscation functions. |
| `pillow_check/` | The first CPU check in Section V: a 10-epoch comparison of the notebook's blur with Pillow's, as Privacy Blur used it, and a bit-exact NumPy version of Pillow's blur with the checks that split the difference into edges, kernel shape and rounding. See its README. |
| `grid_check/` | The second CPU check in Section V: the model trained on the padded 32×32 grid of McPherson et al. and on the paper's 7×7 blocks, seeds 0 to 4, tested after epochs 2 and 10. See its README. |
| `paper_numbers.py`, `paper_numbers.txt` | Prints the numbers in the paper that the notebook does not print, such as the exact means behind three rounding ties, the prediction interval in Section V and the MNIST digit counts, and what it printed. |
| `requirements-cpu-checks.txt` | The pinned Python packages for the CPU checks, `paper_numbers.py` and the figure scripts. |

### Earlier runs

These runs used three seeds without deterministic kernels, and their CSVs give the population
SD, which divides by n = 3. Seeds 0 to 2 of the ten-seed run agree with `results_2026-10-21.csv`
within 0.71 points (0.20 for the model trained on obfuscated images).

| File | What it is |
|---|---|
| `dl_obfuscation.ipynb` | The earlier main experiment: 9 conditions, 2 epochs, seeds 0, 1 and 2, with the printed outputs of its 2026-10-21 run. It writes `results.csv`, `fig_examples.png` and `fig_accuracy.png`. |
| `results_2026-10-21.csv` | Results of that run (Google Colab, NVIDIA T4), formerly `results.csv`. |
| `results_first_run.csv` | An earlier full run of the same notebook (2026-10-07, also a T4), within 0.87 points of `results_2026-10-21.csv` (0.18 for the model trained on obfuscated images). |
| `dl_obfuscation_10epochs.ipynb`, `results_10epochs.csv` | The earlier 10-epoch check on clean, pixelate 7×7 and blur σ = 4, with its printed outputs. |

## Run it

Open `dl_obfuscation_v2.ipynb` in Google Colab and choose a T4 GPU runtime (Runtime → Change
runtime type). Colab already has PyTorch, torchvision, NumPy, pandas, SciPy and Matplotlib.

To rerun only the analysis, upload `results_per_seed.csv` next to the notebook (Files panel →
Upload) and choose Runtime → Run all. With `RUN_TRAINING = False`, the default, nothing is
trained.

For a new run, start a fresh session, set `RUN_TRAINING = True` in the first cell and choose
Runtime → Run all. Training takes about 25 minutes on a T4. A GPU run uses ten seeds and a CPU
run one. On a T4 with the software listed in `run_info.json`, the run should reproduce
`results_per_seed.csv` exactly; other GPUs or versions give slightly different numbers.

To redraw the figures locally you need NumPy and Matplotlib, plus SciPy for Fig. 2 and the
MNIST test images for Fig. 1, which torchvision downloads to `data/MNIST/raw/` when the
notebook runs:

```bash
python make_fig_examples.py data/MNIST/raw/t10k-images-idx3-ubyte.gz fig_examples.png
python make_fig_accuracy.py results_per_seed.csv fig_accuracy.png
```

The CPU checks, `paper_numbers.py` and the figure scripts run locally in the Python environment
that `requirements-cpu-checks.txt` pins; its first lines show how to create it as `.venv`. The
READMEs of `pillow_check/` and `grid_check/` give the commands for the checks, and this prints
the numbers in the paper that the notebook does not print:

```bash
.venv/bin/python paper_numbers.py results_per_seed.csv results_2026-10-21.csv data
```

## Results

The table gives test accuracy in percent, as the mean ± SD over seeds 0 to 9 of the run in
`dl_obfuscation_v2.ipynb`. The SD is the sample SD, which divides by n − 1 = 9, and a 95%
confidence interval for each mean is the mean ± 0.72 SD; the notebook prints every interval,
along with paired differences between conditions, models and epochs. The earlier three-seed
runs report the population SD, which divides by n = 3; the sample SD of the same three values
is 1.22 times larger.

| Condition | Clean-trained, 2 epochs | Obfuscation-trained, 2 epochs | Clean-trained, 10 epochs | Obfuscation-trained, 10 epochs |
|---|---|---|---|---|
| clean | 98.62 ± 0.09 | 98.62 ± 0.09 | 99.19 ± 0.05 | 99.19 ± 0.05 |
| pixelate 2x2 | 98.12 ± 0.26 | 98.39 ± 0.14 | 98.77 ± 0.11 | 99.04 ± 0.06 |
| pixelate 4x4 | 82.24 ± 4.21 | 95.96 ± 0.12 | 79.30 ± 2.14 | 97.35 ± 0.10 |
| pixelate 7x7 | 35.34 ± 3.79 | 83.98 ± 0.64 | 33.79 ± 2.69 | 87.59 ± 0.36 |
| pixelate 14x14 | 13.36 ± 3.07 | 51.15 ± 0.40 | 13.49 ± 3.07 | 51.92 ± 0.26 |
| blur sigma=1 | 97.97 ± 0.23 | 98.19 ± 0.11 | 98.50 ± 0.29 | 99.01 ± 0.07 |
| blur sigma=2 | 89.29 ± 3.89 | 96.35 ± 0.23 | 86.22 ± 7.10 | 98.28 ± 0.18 |
| blur sigma=4 | 36.40 ± 11.00 | 89.28 ± 0.69 | 31.11 ± 6.53 | 94.86 ± 0.18 |
| solid mask | 10.79 ± 0.93 | 11.35 ± 0.00 | 10.78 ± 0.75 | 11.35 ± 0.00 |

## AI use

Claude (Anthropic), an AI assistant, wrote the experiment code: the three notebooks,
`make_fig_examples.py`, `make_fig_accuracy.py`, the scripts in `pillow_check/` and
`grid_check/`, and `paper_numbers.py`. It suggested the outline of the paper and drafted an
earlier version of its text from Shashwat Deep's instructions, which Shashwat Deep rewrote.
Claude also prepared this repository: the README, `CITATION.cff`, the release notes, the PDF,
the commit and the release. A review of an earlier draft by Claude found errors that this
version corrects; Shashwat Deep checked each of them against its source.

Shashwat Deep planned the obfuscation levels, ran the notebooks on Google Colab, ran the CPU
checks, `paper_numbers.py` and the figure scripts on a laptop, and checked the results.

## Citation

S. Deep, "Can a small convolutional network still read blurred and pixelated digits? A
replication of McPherson et al. (2016)," version 1.0.0, Zenodo, Oct. 2026, doi:
10.5281/zenodo.23692156.

```bibtex
@misc{deep2026obfuscation,
  author    = {Shashwat Deep},
  title     = {Can a Small Convolutional Network Still Read Blurred and Pixelated
               Digits? {A} Replication of {McPherson} et al. (2016)},
  year      = {2026},
  month     = oct,
  howpublished = {Zenodo},
  note      = {Version 1.0.0},
  doi       = {10.5281/zenodo.23692156},
  url       = {https://doi.org/10.5281/zenodo.23692156}
}
```

This DOI identifies version 1.0.0, the version the paper prints on its first page. The
Zenodo record also gives a DOI that covers all versions.

## License

The code and results are under the MIT license; see `LICENSE`. The paper (`paper.pdf`) is
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The one exception is
`pillow_check/src_libImaging_BoxBlur.c`, a copy of Pillow's source under Pillow's MIT-CMU
license (`pillow_check/LICENSE-Pillow.txt`).
