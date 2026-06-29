# -*- coding: utf-8 -*-
"""Step 2 (mp_venv): MediaPipe landmarks for Model B -> LMtr.npy. Aligned to Xtr.npy."""
import time, numpy as np, cv2
import mediapipe as mp
try:
    H = mp.solutions.hands.Hands
except AttributeError:
    from mediapipe.python.solutions import hands as _h; H = _h.Hands

IMG_SIZE = 64
Xtr = np.load("Xtr.npy")
hi = H(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3, model_complexity=1)

def _skin(u8):
    e = cv2.createCLAHE(3.0, (4, 4)).apply(u8)
    if e.mean() < 100: e = cv2.bitwise_not(e)
    f = e.astype(np.float32) / 255
    return np.stack([np.clip(f*210+40,0,255), np.clip(f*170+25,0,255), np.clip(f*140+10,0,255)], -1).astype(np.uint8)

def landmarks(img):
    u8 = ((img[:, :, 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)
    p = int(IMG_SIZE * 0.15)
    u8 = cv2.copyMakeBorder(u8, p, p, p, p, cv2.BORDER_CONSTANT, value=int(u8.mean()))
    res = hi.process(_skin(cv2.resize(u8, (256, 256), interpolation=cv2.INTER_LANCZOS4)))
    if res.multi_hand_landmarks:
        return np.array([[q.x, q.y, q.z] for q in res.multi_hand_landmarks[0].landmark], np.float32).flatten()
    return np.zeros(63, np.float32)

t0 = time.time()
LM = np.stack([landmarks(x) for x in Xtr]).astype(np.float32)
hi.close()
np.save("LMtr.npy", LM)
print(f"saved LMtr{LM.shape} | detection rate {LM.any(1).mean():.2%} | {time.time()-t0:.1f}s")
