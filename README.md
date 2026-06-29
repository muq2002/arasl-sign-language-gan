# Arabic Sign Language Image Generation (ArASL 54K)

Two conditional GANs that generate 128×128 grayscale Arabic Sign Language hand images
(32 letter classes) from the **ArASL 54K** dataset.

## Models

| Model | Notebook | Structural supervision | Extra dependency |
|-------|----------|------------------------|------------------|
| **A** | [`notebooks/model_A_cgan_128_no_mediapipe.ipynb`](notebooks/model_A_cgan_128_no_mediapipe.ipynb) | Pixel L1 only | — |
| **B** | [`notebooks/model_B_cgan_128_mediapipe.ipynb`](notebooks/model_B_cgan_128_mediapipe.ipynb) | Pixel L1 + MediaPipe landmark MSE | MediaPipe Hands |

Both share the same backbone: a class-conditional SAGAN generator (self-attention at 32×32),
a spectral-normalized discriminator with spatial label projection, asymmetric learning rates,
label smoothing, and an adaptive G:D update ratio.

## Documentation

An attractive step-by-step walkthrough of how both models work:

➡️ **Open [`docs/index.html`](docs/index.html) in a browser.**

It covers the shared backbone, each model's training steps, the loss functions,
an A-vs-B comparison, the evaluation suite, and honest engineering notes.

## Project structure

```
.
├── README.md
├── docs/             # HTML walkthroughs (models, generator, optimizations, problems, Model C)
├── notebooks/        # A, B, optimized A/B, and the A/B/C comparison notebooks
├── src/              # optimized, modular reimplementation (config, models, data, training)
├── reports/          # literature search + Model C prior-art + verdict/opinion (HTML + md)
├── experiments/      # real local A/B/C runs
│   ├── scripts/          # data prep, MediaPipe landmarks, train + evaluate
│   ├── arrays/           # prepared .npy (gitignored)
│   ├── results/          # metrics JSON + logs
│   └── visualizations/   # HTML report + image grids
└── data/             # dataset (gitignored — large): ArASL_dataset/, samples/, arasl.parquet
```

## A third model — Model C (structure-conditioned)

Beyond the two original models, this repo evaluates **Model C**, which conditions
the generator on a **structure map computed from each image** (Canny edge +
silhouette + distance transform) instead of using MediaPipe as a loss. In the local
5K runs it wins decisively (**GAN-test 0.95 vs ~0.5 for A/B**) and a **held-out
structure test** confirms it generalizes (recognition 0.93 on unseen structures,
generalization gap 0.024, fidelity SSIM 0.95). See [`reports/`](reports/) for the
verdict, prior-art search, and visual A-vs-B-vs-C explanation, and
[`experiments/`](experiments/) for the raw runs.

## Dataset

`ArASL_Database_54K_Final` — ~54,000 grayscale hand-sign images across 32 Arabic letter classes.
Not included in this repo (see `.gitignore`); the notebooks expect it on Google Drive at
`/content/drive/MyDrive/ArASL_Database_54K_Final/`.

## Evaluation

FID, SSIM, LPIPS, intra-class diversity (both models) and PKLE — landmark error (Model B only).

## Notes & direction

See the **Honest Engineering Notes** section in `docs/index.html`. Key takeaway: the proposal
(`docs/assets/proposal.jpg`) points toward a **landmark-conditioned diffusion** model — using
structure as a per-image *conditioning input* rather than as a loss against unaligned targets.
