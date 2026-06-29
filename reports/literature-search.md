# Literature search — findings & sources

Deep search across recognition, GAN/diffusion generation, pose/structure
conditioning, landmark-as-loss, and evaluation, focused on whether anyone has
used our two paths. Date of search: 2026-06.

---

## 1. ArASL2018 dataset and what it's used for

The ArSL2018 dataset = 54,049 grayscale 64×64 images, 32 Arabic alphabet signs,
40 participants (Al Khobar, iPhone 6S). Crucially, **published work on this
dataset is almost entirely *recognition* (classification), not generation.**

State-of-the-art recognition reaches ~98–99% (EfficientNet-B2, Vision
Transformers, transfer learning), and augmentation in those papers is
**traditional** (shifts, zoom, flips) — not GAN-based.

- ArASL2018 dataset paper — [Data in Brief / ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340919301283) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6661066/)
- Transfer-learning / transformer recognition (~99%) — [arXiv 2410.00681](https://arxiv.org/html/2410.00681v1)
- Deep learning + XAI recognition — [arXiv 2501.08169](https://arxiv.org/html/2501.08169v1)
- Image-based ArSL recognition (transfer DL) — [Hindawi/Wiley 2023](https://www.hindawi.com/journals/acisc/2023/5195007/tab1/)
- ViT + LIME for ArSL — [ScienceOpen JDR-2024-0092](https://www.scienceopen.com/hosted-document?doi=10.57197/JDR-2024-0092)

**Takeaway:** generating ArASL images with a (c)GAN — Models A/B/C — is an
**under-explored niche**; the dataset is treated as a recognition benchmark.

---

## 2. GAN-based generation / augmentation for sign language

GAN augmentation for sign-language *classification* exists, but mostly for other
languages (Indian SL), and rarely as conditional photo generation of static
alphabet hands.

- Sign Language Recognition using CNN and **CGAN** — [Springer](https://link.springer.com/chapter/10.1007/978-981-19-1012-8_33)
- Indian Sign Language classification + **image augmentation using GAN** — [ResearchGate](https://www.researchgate.net/publication/378571758)
- General principle (GAN augmentation lifts classifier accuracy) — [Data Augmentation in Classification using GAN](https://www.researchgate.net/publication/320821335)

**Takeaway:** the *idea* of GAN augmentation for SL is known, but no prominent
work does cGAN/DCGAN generation specifically on ArASL2018.

---

## 3. Structure / pose-CONDITIONED generation  (our Model C path)

This is a **large, mature field.** Conditioning a generator on a structure map
(edges, skeleton, pose heatmap, segmentation) is the standard way to control
image synthesis — exactly Model C.

- **Pix2pix** — paired edge/sketch → photo translation (the canonical conditioning baseline).
- **PG2 — Pose Guided Person Image Generation** — [arXiv 1705.09368](https://arxiv.org/pdf/1705.09368): condition image + target pose heatmap → output.
- **Soft-Gated Warping-GAN** for pose-guided synthesis — [arXiv 1810.11610](https://arxiv.org/pdf/1810.11610)
- **MM-Hand** — 3D-aware multi-modal *guided hand* generation (pose-preserving hand images) — [arXiv 2010.01158](https://arxiv.org/pdf/2010.01158)
- **ControlNet** (Zhang et al. 2023) — adds edge/pose/segmentation/depth conditioning to diffusion via zero-convs; the basis of modern controllable hand generation — [vllab/controlnet-hands](https://huggingface.co/vllab/controlnet-hands)
- **AttentionHand** — text-driven controllable hand image generation — [arXiv 2407.18034](https://arxiv.org/pdf/2407.18034)

### Sign-language-specific conditioned generation
- **SignGAN / "Signing at Scale"** (CVPR 2022) — skeletal pose → photo-realistic sign video, **with a dedicated hand keypoint loss** — [CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Saunders_Signing_at_Scale_Learning_to_Co-Articulate_Signs_for_Large-Scale_Photo-Realistic_CVPR_2022_paper.pdf)
- **SignDiff** — diffusion model for ASL production, **ControlNet** + body-shape reinforcement — [arXiv 2308.16082](https://arxiv.org/pdf/2308.16082)
- **Stable Signer** — hierarchical SL generative model, ControlNet latent diffusion backbone — [arXiv 2512.04048](https://arxiv.org/pdf/2512.04048)
- **Diversity-Aware SLP via Pose-Encoding VAE** — UNet generator conditioned on 2D pose + VAE features — [arXiv 2405.10423](https://arxiv.org/pdf/2405.10423)
- Diverse signer avatars (manual + non-manual features) — [arXiv 2508.15988](https://arxiv.org/pdf/2508.15988)

**Takeaway:** structure-conditioning (Model C) is **the established, validated
path.** Every serious SL/hand generation system conditions on pose/structure as
an *input*, often with a keypoint loss as an *auxiliary* on top.

---

## 4. Landmark / keypoint as a LOSS  (our Model B path)

Using a landmark detector/regressor to add a "keypoint loss" is a known technique
— **but in the literature it complements conditioning and is used where detection
is reliable** (faces, full-body in scenes), not as the sole structural signal on
tiny cropped grayscale hands.

- **LandmarkGAN** — synthesizing faces from landmarks — [arXiv 2011.00269](https://arxiv.org/pdf/2011.00269)
- **GP-GAN** — landmark-conditioned face synthesis with multi-loss (adv + perceptual + L1) — [arXiv 1710.00962](https://arxiv.org/pdf/1710.00962)
- **FReeNet** — landmark-guided face reenactment — [arXiv 1905.11805](https://arxiv.org/pdf/1905.11805)
- Talking-face with **landmark regressor L2 loss** inside training — [arXiv 1905.03820](https://arxiv.org/pdf/1905.03820)
- Hand keypoint loss (auxiliary, on top of pose conditioning) — SignGAN above.

**Takeaway:** a landmark *loss* is always paired with landmark *conditioning* and
reliable detectors. **No precedent** uses MediaPipe-landmark-loss as the *primary*
driver on static, low-resolution, grayscale alphabet crops (where detection here
was 2–27%). Model B is an **idiosyncratic, unproven** configuration — which is
consistent with why it added nothing in our run.

---

## 5. Evaluation methodology (our recognition metric)

The metric we used — train a classifier on **real** images, test it on
**generated** images — is the standard, named technique:

- **"How good is my GAN?"** (Shmelkov et al., 2018) defines **GAN-train** and
  **GAN-test**; *GAN-test* = classifier trained on real, evaluated on generated
  (measures realism/recognizability). — [arXiv 1807.09499](https://arxiv.org/pdf/1807.09499)
- GAN-train/GAN-test illustration — [ResearchGate fig](https://www.researchgate.net/figure/Illustration-of-GAN-train-and-GAN-test-GAN-train-learns-a-classifier-on-GAN-generated_fig5_336869491)

**Takeaway:** our recognition-accuracy metric is exactly **GAN-test** — a
principled, accepted way to compare generative models, stronger here than FID at
small sample sizes.

---

## 6. MediaPipe Hands — known operating envelope

- MediaPipe Hands = palm detector → 21 3D keypoints, trained on **real RGB hands
  in scenes** — used widely for SL *recognition* from landmarks (e.g. MediaPipe +
  RNN/CNN). [MDPI Electronics](https://www.mdpi.com/2079-9292/11/19/3228)

**Takeaway:** MediaPipe is designed for landmark *extraction for recognition*,
under realistic imaging conditions — not for guiding generation of tightly
cropped grayscale alphabet images, explaining the 2–27% detection we observed.
