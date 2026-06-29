# Conclusions

Drawn from (a) the literature search in [`literature-search.md`](literature-search.md)
and (b) our own real local A/B/C run ([`../local_run/`](../local_run/)).

---

## 1. Did anyone take the same path to get these results?

**The path that worked (Model C — structure conditioning): yes, the whole field
uses it.** Conditioning a generator on a structure map (edges / silhouette / pose
/ skeleton) is the standard, validated approach to controllable image synthesis —
pix2pix, PG2, Soft-Gated Warping-GAN, MM-Hand, ControlNet — and every modern
sign-language generator (SignGAN, SignDiff, Stable Signer, pose-VAE SLP)
conditions on pose/structure. So Model C is **not novel as a concept; it is the
consensus correct method.** Our experiment reproduced that consensus in miniature.

**The path that failed (Model B — MediaPipe landmark *loss* on static low-res
grayscale): no real precedent.** Landmark losses exist, but always (i) alongside
landmark *conditioning* and (ii) where detection is reliable (faces, full bodies).
Using MediaPipe-as-loss as the *primary* structural signal on tightly-cropped
64–128px grayscale alphabet hands — with 2–27% detection — is an idiosyncratic
configuration we did not find in the literature. Its failure in our run is
therefore expected, not surprising.

**Generation on ArASL2018 specifically: rare.** The dataset is overwhelmingly a
*recognition* benchmark (≈99% accuracy with transformers). cGAN/diffusion
*generation* on it is an under-explored niche — a genuine opportunity, but the
*method* that works is already well established elsewhere.

---

## 2. What our own run showed (and how it aligns)

Real CPU run, 4,750 images, 10 classes, 64px, 6 epochs; classifier-on-real
(GAN-test reference) = 0.984, chance = 0.10:

| Model | Recognition (GAN-test) ↑ | Diversity ↑ | Train time |
|-------|:---:|:---:|:---:|
| A — no MediaPipe            | 0.398 | 0.180 | 545 s |
| B — MediaPipe landmark loss | 0.348 | 0.177 | 522 s |
| **C — structure-conditioned** | **0.968** | **0.300** | 830 s |

- **C ≫ A ≈ B**, with C's recognizability (0.968) almost matching real data
  (0.984). This mirrors the literature: conditioning beats unconditioned + loss.
- **B added nothing over A** — MediaPipe detected only **2.06%** of hands, so the
  landmark loss was masked out for ~98% of samples. Exactly the failure mode the
  literature's reliance on reliable detection predicts.
- **C costs ~1.5× the training time** (heavier conditioned encoder + paired
  discriminator) — a fair, expected trade for the large quality gain.

---

## 3. Why A and B collapse (mechanism, confirmed)

Both A and B compute a structural loss between a **randomly-sampled** generated
image and an **unaligned** real target → the minimizer is the per-class *mean*
image (regress-to-mean), which suppresses diversity (A/B diversity ≈ 0.18 vs C
0.30). Model C removes this because its target and its conditioning map come from
the **same** image — restoring correspondence. This is precisely why the field
conditions on structure rather than scoring against unaligned targets.

---

## 4. Recommendations

1. **Adopt structure conditioning (Model C direction).** It is the validated,
   consensus method and won our test decisively. For static ArSL letters,
   **edge/silhouette/distance-transform maps** are ideal: 100% coverage, no
   detector failures, computed for every image.
2. **Drop MediaPipe-as-loss (Model B).** No precedent, ~2–27% detection, no
   measured benefit. If landmarks are ever wanted, use them as a *conditioning
   input* on the detectable subset, never as the sole loss.
3. **Scale C up** to all 32 classes / 128px / more epochs on a GPU, then graduate
   from pix2pix-style cGAN to **ControlNet-style diffusion** (SignDiff / Stable
   Signer are the templates) for higher fidelity.
4. **Keep the GAN-test metric**, and add **KID** (less biased than FID at small N)
   and a per-class breakdown for the full benchmark.
5. **Framing for a paper:** the novelty is *not* "structure conditioning" (known)
   — it is applying it to **ArASL2018 static-letter generation**, an
   under-explored dataset, with a clean A/B/C ablation showing why the
   MediaPipe-loss route fails and the conditioning route succeeds.

---

## 5. One-line conclusion

> We are on the right track: the structure-conditioned path (Model C) that won our
> experiment is the same path the broader hand/sign-language generation literature
> has already converged on — while the MediaPipe-landmark-loss path (Model B) is an
> unproven detour, and our results explain exactly why it doesn't work here.

*Scope caveat: our numbers come from a reduced CPU run (10/32 classes, 64px, 6
epochs, single seed, no FID). They are directional confirmation, not a full
benchmark — see [`../local_run/README.md`](../local_run/README.md).*
