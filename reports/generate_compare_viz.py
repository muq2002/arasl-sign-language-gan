# -*- coding: utf-8 -*-
"""Generate real-image figures explaining Model C (structure conditioning) vs
Model B (MediaPipe landmarks). Run in the mediapipe venv. Reads the extracted
ArASL_dataset/ PNGs; writes annotated panels to reports/assets/."""
import os, glob, json
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
try:
    H = mp.solutions.hands.Hands; mpd = mp.solutions.drawing_utils; HC = mp.solutions.hands.HAND_CONNECTIONS
except AttributeError:
    from mediapipe.python.solutions import hands as _h, drawing_utils as mpd
    H = _h.Hands; HC = _h.HAND_CONNECTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "ArASL_dataset")
OUT = os.path.join(HERE, "assets"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
                     "text.color": "#e6edf3", "axes.edgecolor": "#2a3340"})
CLASSES = ["bb", "ain", "ha", "dal", "waw", "la"]

def load(path, sz=64):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(g, (sz, sz))

# ── structure-map conversions (Model C) ─────────────────────────────────────
def edge(g):  return cv2.Canny(g, 60, 160)
def silh(g):
    _, s = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - s if s.mean() > 127 else s
def dist(g):  return cv2.normalize(cv2.distanceTransform(silh(g), cv2.DIST_L2, 3), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ── MediaPipe (Model B) same pipeline as extract_lm ─────────────────────────
hi = H(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3, model_complexity=1)
def skin(u8):
    e = cv2.createCLAHE(3.0, (4, 4)).apply(u8)
    if e.mean() < 100: e = cv2.bitwise_not(e)
    f = e.astype(np.float32) / 255
    return np.stack([np.clip(f*210+40,0,255), np.clip(f*170+25,0,255), np.clip(f*140+10,0,255)], -1).astype(np.uint8)
def mp_try(g):
    p = int(64*0.15); u = cv2.copyMakeBorder(g, p, p, p, p, cv2.BORDER_CONSTANT, value=int(g.mean()))
    rgb = skin(cv2.resize(u, (256, 256), interpolation=cv2.INTER_LANCZOS4))
    res = hi.process(rgb)
    if res.multi_hand_landmarks:
        out = rgb.copy(); mpd.draw_landmarks(out, res.multi_hand_landmarks[0], HC)
        return True, out
    return False, rgb

# ════════ FIGURE 1 — Model C structure-map conversions (always works) ════════
fig, ax = plt.subplots(len(CLASSES), 4, figsize=(8, 1.9*len(CLASSES)))
cols = ["real image", "1) Canny edge", "2) silhouette", "3) distance map"]
for r, cls in enumerate(CLASSES):
    f = sorted(glob.glob(os.path.join(DATA, cls, "*.png")))[0]
    g = load(f)
    for c, im in enumerate([g, edge(g), silh(g), dist(g)]):
        ax[r, c].imshow(im, cmap="gray", vmin=0, vmax=255); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0: ax[r, c].set_title(cols[c], fontsize=10, color="#3dba8c" if c else "#9d8df5")
    ax[r, 0].set_ylabel(cls, fontsize=10, color="#9aa7b4", rotation=0, labelpad=18, va="center")
fig.suptitle("Model C — structure maps computed from every image (100% coverage)", color="#3dba8c", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "c_pipeline.png"), dpi=130, bbox_inches="tight"); plt.close()

# ════════ FIGURE 2 — Model B MediaPipe attempts (mostly fails) ════════
scan = []
for cls in CLASSES:
    for f in sorted(glob.glob(os.path.join(DATA, cls, "*.png")))[:120]:
        scan.append((cls, f))
det, nodet, n_det = [], [], 0
for cls, f in scan:
    ok, vis = mp_try(load(f))
    if ok: n_det += 1; det.append((cls, vis))
    else:  nodet.append((cls, vis))
rate = n_det / len(scan)
panel = (det[:2] + nodet)[:6]
while len(panel) < 6: panel.append(nodet[len(panel)])
fig, ax = plt.subplots(2, 3, figsize=(9, 6.2))
for i, a in enumerate(ax.flat):
    cls, vis = panel[i]
    is_det = i < len(det[:2])
    a.imshow(vis); a.set_xticks([]); a.set_yticks([])
    a.set_title(("DETECTED  ("+cls+")" if is_det else "NO HAND DETECTED  ("+cls+")"),
                fontsize=10, color=("#3dba8c" if is_det else "#f85149"))
    for s in a.spines.values(): s.set_color("#3dba8c" if is_det else "#f85149"); s.set_linewidth(2)
fig.suptitle(f"Model B — MediaPipe on the same data: only {rate*100:.1f}% of hands detected",
             color="#e8a450", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "b_mediapipe.png"), dpi=130, bbox_inches="tight"); plt.close()
hi.close()

# ════════ FIGURE 3 — coverage bar ════════
fig, a = plt.subplots(figsize=(7, 2.6))
bars = a.barh(["Model C\n(structure maps)", "Model B\n(MediaPipe)"], [100, rate*100],
              color=["#3dba8c", "#e8a450"])
a.set_xlim(0, 100); a.set_xlabel("usable structural signal (% of images)", color="#9aa7b4")
for b, v in zip(bars, [100, rate*100]):
    a.text(min(v+2, 92), b.get_y()+b.get_height()/2, f"{v:.1f}%", va="center", color="#e6edf3", fontsize=12, fontweight="bold")
a.tick_params(colors="#9aa7b4")
fig.suptitle("Why C beats B: signal coverage", color="#e6edf3", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "coverage.png"), dpi=130, bbox_inches="tight"); plt.close()

json.dump({"mediapipe_detection_rate": round(rate, 4), "scanned": len(scan),
           "structure_coverage": 1.0}, open(os.path.join(OUT, "_viz_meta.json"), "w"), indent=2)
print(f"DONE. MediaPipe detected {n_det}/{len(scan)} ({rate*100:.1f}%). Wrote 3 PNGs to {OUT}")
