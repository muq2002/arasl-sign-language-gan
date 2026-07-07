"""
MediaPipe landmark extraction (Model B) — same 4-strategy cascade and same
colorization as the original, but MUCH faster.

KEY SPEED FIX (accuracy-neutral):
    The original created a fresh `mp.solutions.hands.Hands(...)` object on EVERY
    call and EVERY strategy -> up to ~4 model loads per image -> ~200k model
    constructions over 54k images. Here we build ONE Hands instance per
    confidence level and reuse it for every image. Identical detector, identical
    parameters, identical outputs -- only the wasteful re-instantiation is gone.
"""
import cv2
import numpy as np

from config import (MP_DETECT_SIZE, MP_DETECT_SIZE_FB, MP_CONFIDENCE,
                    MP_CONFIDENCE_LOW, MP_MODEL_COMPLEX)

# NOTE: `mediapipe` is imported lazily (inside HandLandmarkExtractor.__init__)
# so this module can be imported in the TensorFlow env — where mediapipe is NOT
# installed (protobuf 4.x vs TF's protobuf 7.x conflict). With a warm landmark
# cache, compute_landmarks() never constructs an extractor, so mediapipe is
# never imported and training runs fine in the TF env.


# ── colorization helpers (unchanged) ──────────────────────────────────────
def _gray_to_skintone(u8):
    if u8.ndim == 3:
        u8 = u8[:, :, 0]
    f = u8.astype(np.float32) / 255.0
    r = np.clip(f * 210 + 40, 0, 255).astype(np.uint8)
    g = np.clip(f * 170 + 25, 0, 255).astype(np.uint8)
    b = np.clip(f * 140 + 10, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _gray_to_clahe_skintone(u8):
    if u8.ndim == 3:
        u8 = u8[:, :, 0]
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(u8)
    if enhanced.mean() < 100:
        enhanced = cv2.bitwise_not(enhanced)
    return _gray_to_skintone(enhanced)


def _gray_to_clahe_only(u8):
    if u8.ndim == 3:
        u8 = u8[:, :, 0]
    enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(u8)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def _preprocess(u8, target, colorize_fn, pad_ratio=0.15):
    if u8.ndim == 3:
        u8 = u8[:, :, 0]
    pad = int(max(u8.shape) * pad_ratio)
    bg = int(u8.mean())
    padded = cv2.copyMakeBorder(u8, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)
    up = cv2.resize(padded, (target, target), interpolation=cv2.INTER_LANCZOS4)
    return colorize_fn(up)


def _norm_to_u8(img_norm):
    return ((img_norm[:, :, 0] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


class HandLandmarkExtractor:
    """Reusable extractor holding persistent Hands instances (the speed fix).

    Use as a context manager so MediaPipe resources are released:
        with HandLandmarkExtractor() as ext:
            lm = ext.extract(img_norm)
    """

    def __init__(self):
        import mediapipe as mp  # lazy: only needed when actually extracting
        H = mp.solutions.hands.Hands
        # one instance per confidence threshold, reused for all images
        self._hi = H(static_image_mode=True, max_num_hands=1,
                     min_detection_confidence=MP_CONFIDENCE,
                     model_complexity=MP_MODEL_COMPLEX)
        self._lo = H(static_image_mode=True, max_num_hands=1,
                     min_detection_confidence=MP_CONFIDENCE_LOW,
                     model_complexity=MP_MODEL_COMPLEX)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._hi.close()
        self._lo.close()

    @staticmethod
    def _read(res):
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
        return None

    def extract(self, img_norm):
        """4-strategy cascade. Returns float32 (63,); zeros on total failure."""
        u8 = _norm_to_u8(img_norm)

        r = self._read(self._hi.process(_preprocess(u8, MP_DETECT_SIZE, _gray_to_clahe_skintone)))
        if r is not None:
            return r
        r = self._read(self._hi.process(_preprocess(u8, MP_DETECT_SIZE_FB, _gray_to_clahe_skintone)))
        if r is not None:
            return r
        r = self._read(self._lo.process(_preprocess(u8, MP_DETECT_SIZE, _gray_to_clahe_only)))
        if r is not None:
            return r
        raw = cv2.resize(u8, (MP_DETECT_SIZE, MP_DETECT_SIZE), interpolation=cv2.INTER_LANCZOS4)
        r = self._read(self._lo.process(_gray_to_clahe_skintone(raw)))
        return r if r is not None else np.zeros(63, dtype=np.float32)

    def extract_eval(self, img_norm):
        """Returns None on failure (for paired metrics like PKLE)."""
        lm = self.extract(img_norm)
        return lm if lm.any() else None


def compute_landmarks(images, cache_path):
    """Build/load the landmark cache with a single reused extractor."""
    import os
    from tqdm import tqdm
    n = len(images)
    if os.path.exists(cache_path):
        lm = np.load(cache_path)
        if lm.shape == (n, 63):
            pct = lm.any(axis=1).mean() * 100
            # A correctly-shaped cache is authoritative. On ArASL the true
            # MediaPipe detection rate is ~2% (low-res grayscale signs), so the
            # old <20% "stale -> recompute" heuristic would wrongly discard a
            # valid cache and force a mediapipe import in the TF env. Trust it.
            print(f"Cache loaded {lm.shape}  detections {pct:.1f}% (using cache)")
            return lm
        print(f"Cache shape {lm.shape} != {(n, 63)} -> recomputing")
        os.remove(cache_path)

    print(f"Extracting landmarks for {n:,} images (reused MediaPipe instances)...")
    out = np.zeros((n, 63), dtype=np.float32)
    with HandLandmarkExtractor() as ext:
        for i in tqdm(range(n), desc="MediaPipe"):
            out[i] = ext.extract(images[i])
    np.save(cache_path, out)
    print(f"Saved {cache_path}  detections {out.any(axis=1).mean()*100:.1f}%")
    return out
