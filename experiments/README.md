# Experiments — local A/B/C (+ D) comparison (real runs)

Real, **reduced** head-to-head of the GAN models A/B/C — plus the diffusion model
**D** as a separate script ([`scripts/train_eval_D.py`](scripts/train_eval_D.py)) —
executed on this PC (CPU — TensorFlow can't use the RTX 5060 on native Windows)
against the **same ArASL2018 dataset** Models A/B used
(`pain/ArASL_Database_Grayscale` on Hugging Face — 54,049 images, 32 classes,
64×64 grayscale; class names verified identical).

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

## 🌀 Model D — structure-conditioned diffusion + classifier-free guidance

A **different paradigm** from the A/B/C GANs (separate script:
[`scripts/train_eval_D.py`](scripts/train_eval_D.py) → `results/results_D.json`). D keeps
Model C's structure-map conditioning but replaces the single-shot generator with an
**iterative denoising U-Net** (DDPM, MSE on the noise), and adds **classifier-free
guidance (CFG)** — the class label is dropped ~10% of training steps so the net learns
conditional + unconditional scores, and a guidance scale `w` at sampling trades diversity
for class accuracy. Sampling uses **DDIM** (30 steps) to stay tractable on CPU.

**Result of the reduced run** (10 classes · 64px · **10 epochs** · 30 DDIM steps ·
classifier-on-real = 0.980, train time 2,199 s). CFG sweep — recognition vs guidance `w`:

| CFG scale `w` | ≈1.0 (no guidance) | 3.0 | 5.0 |
|---|:---:|:---:|:---:|
| GAN-test recognition | **0.819** | 0.788 | 0.769 |
| Diversity | 0.253 | 0.256 | 0.251 |

> **Honest outcome — not a win here.** Model D scored **~0.82**, *below* Model C's
> **0.95**, and CFG did **not** help (higher `w` slightly *lowered* recognition).
> The most likely reason is that diffusion is **undertrained** at 10 epochs / 30
> DDIM steps — it typically needs far more iterations and sampling steps than a GAN
> to reach its quality ceiling, whereas the GANs converge fast at this tiny scale.
> What *did* hold: diffusion trained **stably** (loss 0.100 → 0.034, no collapse, no
> G/D balancing). Treat as a **reduced, directional** run; a fair comparison needs
> more epochs / DDIM steps (ideally on GPU). Raw numbers: `results/results_D.json`.
> (`w`=0 ≡ `w`=1 because the sampler shortcuts `w`=0 to the conditional path.)

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
python scripts/train_eval_D.py       # Model D (diffusion+CFG): results/results_D.json + CFG sweep
# optional: dump the whole dataset to class folders
python scripts/extract_all_images.py # -> ../data/ArASL_dataset/
```

Dataset: `https://huggingface.co/datasets/pain/ArASL_Database_Grayscale`
