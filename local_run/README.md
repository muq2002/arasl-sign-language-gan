# Local A vs B vs C comparison (real run)

A real, **reduced** head-to-head of the three models, executed on a CPU
(no GPU available locally) against the **same ArASL2018 dataset** Model A/B used
(`pain/ArASL_Database_Grayscale` on Hugging Face — 54,049 images, 32 classes,
64×64 grayscale; verified identical class names).

## Result

| Model | Recognition Acc ↑ | Diversity ↑ |
|-------|:---:|:---:|
| A — no MediaPipe            | 0.333 | 0.067 |
| B — MediaPipe landmark loss | 0.350 | 0.081 |
| **C — structure-conditioned** | **0.533** | **0.382** |

Classifier sanity on **real** images = 0.96 · random chance = 0.33

**Takeaways (consistent with the design analysis):**
- **A** sits at chance with collapsed diversity → regressed to the per-class mean.
- **B ≈ A**: MediaPipe detection was only **1.6%** at 64px, so the landmark loss
  was masked out for ~98% of samples and contributed almost nothing.
- **C wins both metrics**: structure maps cover **100%** of images and give the
  generator a real per-image target → more recognizable *and* ~5× more diverse.

## ⚠️ Scope / honesty

This is a **toy-scale** run, NOT the full benchmark:
- 3 classes (of 32), 150 images/class, **64×64** native, **6 epochs**, single seed, CPU.
- FID/LPIPS omitted (need PyTorch); metrics are recognition accuracy + diversity.

Treat it as a **directional signal**, not a final result. It confirms the
*direction* (C > A ≈ B) but the magnitudes will change at full scale.
For the full benchmark use `notebooks/model_comparison_ABC.ipynb` on a GPU.

## Reproduce

Two isolated venvs are used because mediapipe (protobuf 4.x) and modern
TensorFlow (protobuf 6.x) cannot share one environment.

```bash
# main venv: tensorflow, opencv, pandas, pyarrow, scikit-learn, pillow
python prep_data.py        # download arasl.parquet from HF first; builds Xtr/Ctr/...
# mediapipe venv: mediapipe==0.10.14, opencv, numpy
python extract_lm.py       # MediaPipe landmarks -> LMtr.npy
# main venv:
python train_eval.py       # trains A/B/C, prints the table, writes local_results.json
```

Dataset: `https://huggingface.co/datasets/pain/ArASL_Database_Grayscale`
(single parquet at `data/train-00000-of-00001-*.parquet`).
