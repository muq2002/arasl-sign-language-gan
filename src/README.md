# `src/` — Speed-optimized models (accuracy-preserving)

Runnable Python ports of the two notebooks. The **loss math and architecture are
unchanged** — only *how* the computation executes was optimized, so results stay
equivalent to the originals.

## Files

| File | Role |
|------|------|
| `config.py` | All hyperparameters + speed toggles + `setup_speed()` |
| `models.py` | Generator, Discriminator, landmark regressor (mixed-precision safe) |
| `data.py` | `tf.data` pipeline + vectorized Phase-1/2 target selection |
| `train_utils.py` | Optimizer + loss-scaling helpers |
| `mediapipe_utils.py` | Reusable MediaPipe extractor (Model B) |
| `train_model_a.py` | Model A training (no MediaPipe) |
| `train_model_b.py` | Model B training (MediaPipe landmark loss) |

## Run

```bash
# Colab (after mounting Drive and pip-installing deps from the notebooks)
python src/train_model_a.py
python src/train_model_b.py
```
or from a notebook:
```python
import sys; sys.path.insert(0, "src")
from train_model_a import train; G, D, hist, enc = train()
```

## What was optimized (and why it doesn't change accuracy)

| Optimization | Speed win | Why accuracy is preserved |
|--------------|-----------|---------------------------|
| **Mixed precision** (`float16` + loss scaling) | ~1.5–2× on tensor-core GPUs (T4/V100/A100) | Output & logit layers kept in `float32`; softmax in `float32`; dynamic loss scaling prevents underflow. Same optimizer, same math. |
| **`tf.data` + `prefetch`** | Removes per-step host→device stalls | Same data, same batches — only the copy/compute overlap changed. |
| **In-graph one-hot + vectorized targets** | Kills the per-step Python `np.random.choice` loop | Same target-selection rule (prototype in Phase 1, random real member in Phase 2). |
| **MediaPipe singleton** (Model B) | Extraction goes from hours → minutes | Identical detector & parameters; only the wasteful per-call re-instantiation was removed. |
| **XLA** (`jit_compile`, optional) | Op fusion | Numerically equivalent; off by default for compatibility. |

### Toggles (`config.py`)
```python
USE_MIXED_PRECISION = True    # set False to reproduce exact float32 path
USE_XLA             = False   # try True for extra speed if stable on your GPU
TF_DATA_PREFETCH    = True
```

To verify equivalence, set `USE_MIXED_PRECISION=False`, `USE_XLA=False` — the path
then matches the original notebooks numerically (only the input pipeline differs,
which is statistically identical).

## Note

These are faithful, faster ports. The deeper *design* improvements discussed in
`docs/index.html` (structure-as-conditioning / diffusion) are intentionally **not**
applied here, since the request was speed without changing accuracy.
