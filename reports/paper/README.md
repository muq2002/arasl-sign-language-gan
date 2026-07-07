# Paper deliverables — ArASL Sign-Language Generation (Models A, B, C)

This folder collects everything needed to write the paper: full-resolution
(128×128) GPU training results, loss curves, evaluation metrics, sample grids,
model cards, and an end-user inference interface.

Generated on the local machine (WSL2 + RTX 3050, TensorFlow 2.21 GPU).

## Contents (populated by the pipeline)

| Path | What |
|------|------|
| `charts/`        | Loss curves + metric comparison bar charts (PNG, 150–300 dpi) |
| `figures/`       | Sample grids: real vs A / B / C, held-out structure test |
| `results/`       | `metrics.json`, `metrics.csv` — recognition, diversity, SSIM, FID |
| `models/`        | Exported generators (`*.keras`) + label maps for the interface |
| `model_cards.md` | Per-model architecture, hyperparameters, training notes |
| `paper.md`       | Paper-ready writeup (abstract, method, results, discussion) |

## How it was produced

1. `src/train_model_a.py`, `src/train_model_b.py`, `src/train_model_c.py`
   — full 50-epoch 128px training on the RTX 3050 (mixed precision).
2. `src/export_models.py` — exports inference-ready generators.
3. `src/paper_eval.py` — one reference classifier on real held-out images;
   computes GAN-test recognition, diversity, SSIM (+ FID if torch present),
   and Model C's held-out structure-generalization test.
4. `src/paper_charts.py` — renders all charts and sample grids here.
5. `interface/app.py` — end-user demo: pick a letter → generate a hand sign.
