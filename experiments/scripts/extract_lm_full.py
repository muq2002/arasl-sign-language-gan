"""
Full 128px MediaPipe landmark extraction for Model B (run in the MediaPipe env,
which has NO TensorFlow). Reads the image array dumped by src/dump_images_for_lm.py
and writes landmarks_128px.npy that train_model_b.py caches.

4-strategy detection cascade mirrors src/mediapipe_utils.py (reused Hands
instances for speed). Detection on ArASL is intrinsically low (~2-5%) because the
signs are low-res grayscale; that is the point of Model B.

Run (mp env):  python experiments/scripts/extract_lm_full.py [BASE_DIR]
BASE_DIR defaults to <repo>/outputs/cgan_B_128mp
"""
import os, sys, time, json
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

IMG_SIZE = 128
MP_DETECT_SIZE, MP_DETECT_SIZE_FB = 256, 320
MP_CONFIDENCE, MP_CONFIDENCE_LOW, MP_MODEL_COMPLEX = 0.30, 0.15, 1

_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_repo, "outputs", "cgan_B_128mp")
IMAGES = os.environ.get("ARASL_LM_IMAGES", "/root/arasl_b_images_128.npy")
OUT = os.path.join(BASE, "landmarks_128px.npy")
PROG = OUT + ".progress"           # resume marker (# images completed)


def _skintone(u8):
    e = cv2.createCLAHE(3.0, (4, 4)).apply(u8)
    if e.mean() < 100:
        e = cv2.bitwise_not(e)
    f = e.astype(np.float32) / 255.0
    return np.stack([np.clip(f * 210 + 40, 0, 255), np.clip(f * 170 + 25, 0, 255),
                     np.clip(f * 140 + 10, 0, 255)], -1).astype(np.uint8)


def _clahe_only(u8):
    return cv2.cvtColor(cv2.createCLAHE(4.0, (4, 4)).apply(u8), cv2.COLOR_GRAY2RGB)


def _prep(u8, target, fn, pad_ratio=0.15):
    pad = int(max(u8.shape) * pad_ratio)
    bg = int(u8.mean())
    padded = cv2.copyMakeBorder(u8, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)
    return fn(cv2.resize(padded, (target, target), interpolation=cv2.INTER_LANCZOS4))


def _read(res):
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lm], np.float32).flatten()
    return None


def main():
    if not os.path.exists(IMAGES):
        raise SystemExit(f"missing {IMAGES} - run src/dump_images_for_lm.py first")
    imgs = np.load(IMAGES).astype(np.float32)  # (N,128,128,1) in [-1,1]
    n = len(imgs)
    print(f"extracting landmarks for {n:,} images from {IMAGES}")
    H = mp.solutions.hands.Hands
    hi = H(static_image_mode=True, max_num_hands=1, min_detection_confidence=MP_CONFIDENCE, model_complexity=MP_MODEL_COMPLEX)
    lo = H(static_image_mode=True, max_num_hands=1, min_detection_confidence=MP_CONFIDENCE_LOW, model_complexity=MP_MODEL_COMPLEX)
    # resume from a partial run if present (WSL can restart mid-extraction)
    start = 0
    if os.path.exists(OUT) and os.path.exists(PROG):
        prev = np.load(OUT)
        if prev.shape == (n, 63):
            out = prev
            try:
                start = int(open(PROG).read().strip())
            except Exception:
                start = 0
            print(f"resuming from image {start}/{n}")
        else:
            out = np.zeros((n, 63), np.float32)
    else:
        out = np.zeros((n, 63), np.float32)
    t0 = time.time()
    for i in tqdm(range(start, n), desc="MediaPipe", initial=start, total=n):
        u8 = ((imgs[i, :, :, 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)
        r = _read(hi.process(_prep(u8, MP_DETECT_SIZE, _skintone)))
        if r is None:
            r = _read(hi.process(_prep(u8, MP_DETECT_SIZE_FB, _skintone)))
        if r is None:
            r = _read(lo.process(_prep(u8, MP_DETECT_SIZE, _clahe_only)))
        if r is None:
            raw = cv2.resize(u8, (MP_DETECT_SIZE, MP_DETECT_SIZE), interpolation=cv2.INTER_LANCZOS4)
            r = _read(lo.process(_skintone(raw)))
        if r is not None:
            out[i] = r
        if (i + 1) % 5000 == 0:                     # checkpoint for resume
            np.save(OUT, out)
            open(PROG, "w").write(str(i + 1))
    hi.close(); lo.close()
    dt = time.time() - t0
    np.save(OUT, out)
    rate = float(out.any(1).mean())
    json.dump({"seconds": round(dt, 1), "detection_rate": round(rate, 4), "n": n},
              open(os.path.join(BASE, "lm_full_meta.json"), "w"), indent=2)
    print(f"saved {OUT}  detection {rate:.2%}  in {dt:.0f}s")


if __name__ == "__main__":
    main()
