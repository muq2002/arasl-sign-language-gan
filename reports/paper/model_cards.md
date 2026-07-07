# Model Cards — ArASL Sign-Language Generation

Dataset: **ArASL 54K** (`pain/ArASL_Database_Grayscale`, 54,049 grayscale hand
images, 32 Arabic-letter classes). Full runs: **128×128**, mixed precision
(float16 compute + loss scaling), RTX 3050 (WSL2), TensorFlow 2.21.

Shared training schedule (`src/config.py`): Adam (β1=0.5, clipnorm=1.0),
`LR_G=2e-4`, `LR_D=1e-4` (asymmetric), label smoothing 0.9, 50 epochs, batch 32,
LR halved at epoch 20 (D) / 35 (G). Latent `Z_DIM=128`.

---

## Model A — class-conditioned cGAN, pixel loss only (baseline)

- **Conditioning:** class label + noise only.
- **Generator:** dense → 4×4×512 → 5× Conv2DTranspose (256→128→128→64→32) with
  SAGAN **self-attention at the 3rd upsample block**, BatchNorm+ReLU, tanh
  (float32) output. ~128px.
- **Discriminator:** spatial label projection concatenated to image → 5×
  spectral-normalized Conv2D (64→128→256→512→512), LeakyReLU + dropout, float32
  logit.
- **Loss:** `G = adv + λ_pix(epoch)·mean|fake − target|`, λ_pix warms 0.5→5.0.
  Target is an **unaligned** real image of the class → L1 pulls toward the class
  mean (regress-to-mean → low diversity).
- **Optimizations (accuracy-neutral):** mixed precision, tf.data prefetch,
  in-graph one-hot, vectorized per-epoch target selection, adaptive G:D ratio.

## Model B — A + MediaPipe landmark loss

- **Everything in A**, plus a frozen landmark regressor (63 outputs = 21 joints ×
  xyz) trained on images where MediaPipe Hands detected a hand.
- **Extra loss:** `+ λ_lm(epoch)·mean((reg(fake) − real_lm)² · valid_mask)`,
  λ_lm warms 0→2.0 after a 15-epoch delay; minus a small diversity bonus
  (`−0.05·mean var(fake)`).
- **Reality:** MediaPipe detection on ArASL is ~2% (low-res grayscale), so the
  landmark term is masked out for ~98% of samples → little effect over A. The
  target is still unaligned → same regress-to-mean.

## Model C — structure-conditioned cGAN (the one that works)

- **Conditioning changes, not just the loss.** For every image a 3-channel
  **structure map** is computed with OpenCV — **Canny edges + Otsu silhouette +
  distance transform** (100% coverage, no detector to fail).
- **Generator (encoder–decoder):** encodes the structure map 128→8 (Conv 64→128→
  256→512), fuses with noise + label embedding, decodes 8→128 → image.
- **Discriminator (paired):** judges `(image, structure, label)` triples
  (pix2pix-style), 5× spectral-norm Conv.
- **Loss:** `G = adv + 5.0·mean|fake − real|`, where the L1 target is the **same
  image the structure came from** → spatial correspondence restored → no
  regress-to-mean → diversity recovers.
- **Held-out structure test:** feed C structure maps from images it never trained
  on; small train↔held-out recognition gap ⇒ it learned structure→image, not
  memorization.

---

*Final metrics (recognition / diversity / SSIM / held-out gap) are filled in from
`reports/paper/results/metrics.json` after the full runs complete.*
