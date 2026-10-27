# Pillow blur check

These CPU checks back the paper's statement (Section V) that the blur implementation does not
explain the remaining 2.7-point gap to Privacy Blur. Privacy Blur blurred 8-bit images with
Pillow's `ImageFilter.GaussianBlur`, at a radius of a tenth of "the maximum diagonal". The
paper reads this as the diagonal of the 28×28 digit, a radius of 3.96, and the checks use that
reading; the code of Yang et al., which Privacy Blur cites for the setting, would give 2.8.
Shashwat Deep ran the checks outside the notebook, on a laptop CPU (Intel Core i7-14650HX, 4
threads), with PyTorch 2.14.1 and Pillow 12.3.0; the `.txt` files hold what the summary and
measurement scripts printed.

## Ten-epoch training run

| File | What it is |
|---|---|
| `pil_blur_10ep.py` | Trains the paper's model on obfuscated images for 10 epochs, seeds 0, 1 and 2, under three σ ≈ 4 blurs: the notebook's blur, Pillow's `GaussianBlur(3.96)` on 8-bit images, and the notebook's blur with zero padding instead of reflection. The model, optimizer, batch size and training loop are the notebook's. |
| `pil_blur_10ep.jsonl` | Its nine results: correct answers out of 10,000 for each blur and seed, with the software versions, the CPU and the thread count. |
| `pil_summary.py`, `pil_summary.txt` | Means, 95% t-intervals and paired differences from `pil_blur_10ep.jsonl`, and what it printed. |

Test accuracy after 10 epochs (%), mean over seeds 0 to 2 [95% interval]:

| Blur | Accuracy | Difference from the notebook's blur, paired by seed |
|---|---|---|
| Notebook (reflect padding) | 94.91 [94.75, 95.07] | |
| Pillow `GaussianBlur(3.96)` | 94.42 [94.15, 94.70] | −0.49 [−0.92, −0.05] |
| Notebook with zero padding | 94.18 [93.58, 94.78] | −0.73 [−1.46, −0.01] |

The last line that `pil_summary.py` prints quotes the 10-epoch result of the earlier three-seed
T4 run (`results_10epochs.csv`). The ten-seed run gives 94.86 ± 0.18.

## How Pillow blurs

| File | What it is |
|---|---|
| `emul.py` | A NumPy re-implementation of Pillow 12.3.0's `GaussianBlur` for 8-bit greyscale images: three extended box blurs per axis, edge pixels copied outward at every pass, rounding after every pass. It matches Pillow bit for bit. |
| `t1.py` | Checks `emul.py` against Pillow on random images and MNIST digits, and prints the box radii, weights and kernel SD. |
| `t2.py` | The width of Pillow's kernel, how much of a line next to the border each kind of padding keeps, and the total ink on the MNIST test set. |
| `t3.py` | Splits the difference between the notebook's blur rounded to 8 bits and Pillow's blur into edge handling, kernel shape and rounding. |
| `pil_measure.py` | A separate, quick measurement of the kernel width and the ink figures. |
| `t1.txt`, `t2.txt`, `t3.txt`, `pil_measure.txt` | What `t1.py`, `t2.py`, `t3.py` and `pil_measure.py` printed. In `t1.txt`, no pixel differs between `emul.py` and Pillow. |
| `src_libImaging_BoxBlur.c` | Pillow 12.3.0's `src/libImaging/BoxBlur.c`, which `emul.py` follows. It is Pillow's code, under Pillow's MIT-CMU license (`LICENSE-Pillow.txt`). |

## Run

You need PyTorch, torchvision, NumPy and Pillow, and SciPy for `pil_summary.py`;
`requirements-cpu-checks.txt` in the main folder pins them. `data` below is the MNIST folder that
torchvision uses; `pil_blur_10ep.py` downloads MNIST into it, and the other scripts expect it to be
there already. Run from the main folder:

```bash
.venv/bin/python pillow_check/pil_blur_10ep.py data pillow_check/pil_blur_10ep.jsonl
.venv/bin/python pillow_check/pil_summary.py pillow_check/pil_blur_10ep.jsonl
.venv/bin/python pillow_check/t1.py pillow_check data
.venv/bin/python pillow_check/pil_measure.py data
```

`t2.py` and `t3.py` take the same arguments as `t1.py`. `pil_blur_10ep.py` will not overwrite an
existing output file, and it records the software, the CPU and the thread count with each result;
on the laptop it took about 13 minutes.
