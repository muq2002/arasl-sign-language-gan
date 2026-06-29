# Experiments — local A vs B vs C comparison (real runs)

Real, **reduced** head-to-head of the three models, executed on this PC (CPU —
TensorFlow can't use the RTX 5060 on native Windows) against the **same ArASL2018
dataset** Model A/B used (`pain/ArASL_Database_Grayscale` on Hugging Face —
54,049 images, 32 classes, 64×64 grayscale; class names verified identical).

## 📁 Folder layout

| Folder | What's in it |
|--------|--------------|
| [`scripts/`](scripts/) | the pipeline code (data prep, landmarks, training, eval) |
| `arrays/` | prepared `.npy` arrays + `*_meta.json` *(gitignored — large/local)* |
| [`results/`](results/) | metrics JSON + training logs |
| [`visualizations/`](visualizations/) | HTML report + image grids |

> The **dataset** lives outside this folder, in [`../data/`](../data/)
> (`ArASL_dataset/`, `samples/`, `arasl.parquet` — all gitignored).

## 📊 Visualization

Open **[`visualizations/index.html`](visualizations/index.html)** — metric bars,
training-time comparison, and real-vs-A/B/C sample grids.

## Main result — 5,000-image run

10 classes · 4,750 train · 250 held-out · 64×64 · 6 epochs · CPU
(`results/results_full.json`, reference classifier on real = **0.956**, chance = 0.10)

| Model | GAN-test ↑ | GAN-train ↑ | Diversity ↑ | Train time |
|-------|:---:|:---:|:---:|:---:|
| A — no MediaPipe            | 0.447 | 0.144 | 0.105 | 432 s |
| B — MediaPipe landmark loss | 0.547 | 0.232 | 0.125 | 372 s |
| **C — structure-conditioned** | **0.947** | 0.204 | **0.330** | 889 s |

- **GAN-test** = classifier trained on real, tested on generated (recognizability).
- **GAN-train** = classifier trained on generated, tested on real (usefulness as data).
- **C ≫ A ≈ B** on recognizability and diversity; C is uniformly strong across all
  10 classes (0.90–1.0) while A/B are uneven.
- B's MediaPipe detection was only **2.06%** at 64px → its landmark loss was masked
  out for ~98% of samples and added little.

## ⭐ Held-out structure test (Model C) — does it generalize or just copy?

Feed Model C **structure maps from the 250 held-out images it never trained on**:

| Metric | Value | Meaning |
|--------|:---:|---------|
| Recognition on **held-out** structures | **0.928** | renders unseen structures correctly |
| Recognition on **training** structures | 0.952 | reference |
| **Generalization gap** | **0.024** | tiny → it generalizes, does **not** memorize |
| Fidelity **SSIM**(C output, real target) | **0.950** | faithfully reconstructs the unseen target |

Qualitative grid: `visualizations/assets/heldout_C.png`
(columns = unseen structure → C output → real target).

**Conclusion:** C produces recognizable hands from structures it has never seen,
with only a 2.4-point drop vs training structures — strong evidence it learned the
structure→image mapping rather than copying.

## ⚠️ Scope / honesty

Reduced run, **not** the full benchmark: 10/32 classes, 64×64, 6 epochs, single
seed, CPU, no FID/LPIPS (need PyTorch). Treat magnitudes as **directional**; the
ordering **C ≫ A ≈ B** and the held-out result are the real signals. For the full
GPU benchmark use `notebooks/model_comparison_ABC.ipynb`.

## Reproduce

Two isolated venvs are required (mediapipe needs protobuf 4.x, modern TensorFlow
needs protobuf 6.x — they can't share one env). All scripts resolve their own paths
(`../data`, `arrays/`, `results/`, `visualizations/`), so run them from anywhere:

```bash
# main venv (TF):  tensorflow opencv-python pandas pyarrow scikit-learn pillow
python scripts/prep_data.py          # ../data/arasl.parquet -> arrays/Xtr,Ctr,...
# mediapipe venv:  mediapipe==0.10.14 opencv-python numpy
python scripts/extract_lm.py         # MediaPipe landmarks -> arrays/LMtr.npy
# main venv (TF):
python scripts/train_eval.py         # quick A/B/C: results/results_5k.json + grids
python scripts/train_eval_full.py    # full metrics + held-out test: results/results_full.json
# optional: dump the whole dataset to class folders
python scripts/extract_all_images.py # -> ../data/ArASL_dataset/
```

Dataset: `https://huggingface.co/datasets/pain/ArASL_Database_Grayscale`
