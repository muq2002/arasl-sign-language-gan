# Reports — literature search & conclusions

Investigation of one question: **has anyone taken the same path we took** (the
two structural-supervision strategies for Arabic Sign Language image generation),
and **what does the literature say about which path is correct?**

## Files
- [`model-C-prior-art.md`](model-C-prior-art.md) — **focused answer: has anyone published Model C's exact idea?**
- [`literature-search.md`](literature-search.md) — all search findings, organized by theme, with sources.
- [`conclusions.md`](conclusions.md) — the conclusions drawn from the search + our own local A/B/C run.

## Direct answer: did anyone use Model C's idea?

**Yes — the method, not the exact application.** Conditioning a generator on an
**edge/silhouette structure map** is a published, validated technique
(**ControlNet-Canny**, **Edge-GAN**, **pix2pix** edge→photo), and it has even been
used for **fingerspelling synthesis** (SignON adds a hand-segmentation term to its
GAN; OpenFS does letter-conditioned synthesis). What appears **not yet published**
is Model C's *specific* application — a multi-channel edge+silhouette+distance
condition generating **ArASL2018 Arabic-alphabet** images. So Model C is sound,
proven ground; the novelty is *applying a known method to an under-explored
dataset*, not inventing a new one. See [`model-C-prior-art.md`](model-C-prior-art.md).

## TL;DR

| Our path | Does prior work use it? | Verdict |
|----------|------------------------|---------|
| **Structure/pose CONDITIONING** (Model C: edge/silhouette → image) | **Yes — extensively** (pix2pix, PG2, ControlNet, SignGAN, SignDiff) | The **field-standard, validated** approach |
| **MediaPipe-landmark as a LOSS** on static low-res grayscale alphabet images (Model B) | **No established precedent** for this exact setup | Idiosyncratic; consistent with why it failed for us |
| **Unconditioned cGAN + pixel L1** (Model A) | Common baseline | Known to under-perform conditioning |
| **GAN generation on ArASL2018 specifically** | **Rare** — the dataset is used almost entirely for *recognition* | An under-explored niche |
| **Our metric** (classifier on real → test on generated) | Yes — this is the standard **"GAN-test"** | Principled, correct choice |

**Bottom line:** the path that *won* our experiment (Model C, structure
conditioning) is exactly the path the wider literature has converged on. The path
that *failed* (Model B, MediaPipe-loss) has no real precedent for this kind of
data. Our small local result reproduces, in miniature, the field's consensus.
