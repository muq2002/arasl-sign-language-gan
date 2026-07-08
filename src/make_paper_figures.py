"""
Generate paper figures from REAL ArASL images (CPU-only, no GPU / no TF):
  reports/paper/figures/real_samples_grid.png   — 1 real image per letter, labeled
  reports/paper/figures/structure_illustration.png — real -> Canny | silhouette | distance
These illustrate the dataset and the structure-map conditioning used by C/F/G.
Run:  python src/make_paper_figures.py
"""
import os, glob, json
import numpy as np
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "ArASL_dataset")
FIG = os.path.join(REPO, "reports", "paper", "figures")
os.makedirs(FIG, exist_ok=True)
S = 128
CANNY_LO, CANNY_HI = 60, 160


def letters():
    return sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))


def first_img(letter):
    fs = sorted(glob.glob(os.path.join(DATA, letter, "*.png")) +
                glob.glob(os.path.join(DATA, letter, "*.jpg")))
    im = cv2.imread(fs[0], cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (S, S))


def label_tile(img, text):
    """Return an (S+22)xS 3ch tile: image with a caption bar under it."""
    tile = np.full((S + 22, S, 3), 255, np.uint8)
    tile[:S] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(tile, text, (4, S + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    return tile


def real_grid():
    ls = letters()
    cols = 8
    rows = int(np.ceil(len(ls) / cols))
    pad = 6
    tw, th = S + pad, S + 22 + pad
    canvas = np.full((rows * th + pad, cols * tw + pad, 3), 245, np.uint8)
    for i, lt in enumerate(ls):
        r, c = divmod(i, cols)
        tile = label_tile(first_img(lt), lt)
        y, x = r * th + pad, c * tw + pad
        canvas[y:y + S + 22, x:x + S] = tile
    out = os.path.join(FIG, "real_samples_grid.png")
    cv2.imwrite(out, canvas)
    print("wrote", out, canvas.shape)


def structure_illustration(letter="ain"):
    g = first_img(letter)
    edge = cv2.Canny(g, CANNY_LO, CANNY_HI)
    _, sil = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if sil.mean() > 127:
        sil = 255 - sil
    dist = cv2.normalize(cv2.distanceTransform(sil, cv2.DIST_L2, 3), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    panels = [("real image", g), ("Canny edges", edge), ("silhouette", sil), ("distance transform", dist)]
    pad = 8
    tw = S + pad
    canvas = np.full((S + 26 + pad, len(panels) * tw + pad, 3), 245, np.uint8)
    for i, (name, im) in enumerate(panels):
        x = i * tw + pad
        canvas[pad:pad + S, x:x + S] = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas, name, (x + 2, S + pad + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    out = os.path.join(FIG, "structure_illustration.png")
    cv2.imwrite(out, canvas)
    print("wrote", out, canvas.shape)


if __name__ == "__main__":
    real_grid()
    structure_illustration()
    print("FIGURES_DONE")
