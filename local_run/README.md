# Local A vs B vs C comparison (real runs)

Real, **reduced** head-to-head of the three models, executed on this PC (CPU —
TensorFlow can't use the RTX 5060 on native Windows) against the **same ArASL2018
dataset** Model A/B used (`pain/ArASL_Database_Grayscale` on Hugging Face —
54,049 images, 32 classes, 64×64 grayscale; class names verified identical).

## 📊 Visualization

Open **[`visualizations/index.html`](visualizations/index.html)** — metrics bars,
training-time comparison, and real-vs-A/B/C sample grids.

## Main result (5,000-image run)

10 classes · 4,750 train images · 64×64 · 6 epochs · CPU · classifier on real = **0.984**

| Model | Recognition Acc ↑ | Diversity ↑ | Train time |
|-------|:---:|:---:|:---:|
| A — no MediaPipe            | 0.398 | 0.180 | **545 s** |
| B — MediaPipe landmark loss | 0.348 | 0.177 | **522 s** |
| **C — structure-conditioned** | **0.968** | **0.300** | **830 s** |

Setup costs: MediaPipe extraction 79 s · classifier 349 s · random chance = 0.10

**Takeaways (consistent with the design analysis):**
- **C ≫ A ≈ B.** Model C reaches 0.968 recognition — almost matching real data
  (0.984) — and is the most diverse.
- **B ≈ A**: MediaPipe detection was only **2.06%** at 64px, so its landmark loss
  was masked out for ~98% of samples and added nothing (and cost 79 s extra).
- **C is ~1.5× slower** to train (heavier conditioned encoder + paired
  discriminator) — a fair trade for the large quality jump.

## ⚠️ Scope / honesty

Reduced run, **not** the full benchmark: 10/32 classes, 64×64, 6 epochs, single
seed, CPU, no FID/LPIPS (need PyTorch). Treat magnitudes as **directional**; the
ordering C ≫ A ≈ B is the real signal. For the full GPU benchmark use
`notebooks/model_comparison_ABC.ipynb`.

## Files in this folder

| Path | What |
|------|------|
| `ArASL_dataset/` | all 54,049 images in 32 class folders (gitignored — local only) |
| `dataset_sample/` | 40 quick-look images (gitignored) |
| `arasl.parquet` | source file from Hugging Face (gitignored) |
| `*.npy`, `*_meta.json` | prepared 5K arrays (gitignored) |
| `prep_data.py` | load + subset + structure maps → arrays |
| `extract_lm.py` | MediaPipe landmarks (run in the mediapipe venv) |
| `train_eval.py` | train A/B/C, time them, evaluate, write results + grids |
| `extract_all_images.py` | dump the full parquet to class folders |
| `results_5k.json` | metrics + per-model times |
| `visualizations/` | HTML report + sample-grid PNGs |

## Reproduce

Two isolated venvs are required (mediapipe needs protobuf 4.x, modern TensorFlow
needs protobuf 6.x — they can't share one env):

```bash
# main venv (TF):    tensorflow opencv-python pandas pyarrow scikit-learn pillow
python prep_data.py          # builds Xtr/Ctr/... from arasl.parquet
# mediapipe venv:    mediapipe==0.10.14 opencv-python numpy
python extract_lm.py         # MediaPipe landmarks -> LMtr.npy
# main venv (TF):
python train_eval.py         # trains A/B/C, prints table, writes results_5k.json + viz_out/
```

Dataset: `https://huggingface.co/datasets/pain/ArASL_Database_Grayscale`
