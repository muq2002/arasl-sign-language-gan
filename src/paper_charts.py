"""
Render paper charts from training histories + evaluation metrics into
reports/paper/charts/. Tolerant of missing models (renders what exists).

Reads:
  outputs/cgan_A_128/checkpoints/progress.json   (d, g_adv, g_pixel, g_total, ...)
  outputs/cgan_B_128mp/checkpoints/progress.json (d, g_adv, g_pix, g_lm, ...)
  outputs/cgan_C_128struct/checkpoints/progress.json (d, g, g_adv, g_l1)
  reports/paper/results/metrics.json             (written by paper_eval.py)

Run:  python src/paper_charts.py
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "paper", "charts")
RESULTS = os.path.join(REPO, "reports", "paper", "results", "metrics.json")
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "A": (C.DRIVE_BASE_A, "#d29922", "A — pixel loss only"),
    "B": (C.DRIVE_BASE_B, "#3fb950", "B — + MediaPipe landmark loss"),
    "C": (C.DRIVE_BASE_C, "#a371f7", "C — structure-conditioned"),
}


def _load_hist(base):
    p = os.path.join(base, "checkpoints", "progress.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def chart_per_model_losses():
    for k, (base, color, title) in MODELS.items():
        h = _load_hist(base)
        if not h:
            continue
        ep = range(1, len(h.get("d", [])) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        ax[0].plot(ep, h.get("d", []), label="D loss", color="#f85149")
        gkey = "g_total" if "g_total" in h else "g"
        ax[0].plot(ep, h.get(gkey, []), label="G loss", color=color)
        ax[0].set_title(f"Model {k}: adversarial losses"); ax[0].set_xlabel("epoch")
        ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(ep, h.get("g_adv", []), label="G adversarial", color="#58a6ff")
        if "g_pixel" in h or "g_pix" in h:
            ax[1].plot(ep, h.get("g_pixel", h.get("g_pix", [])), label="G pixel-L1", color="#d29922")
        if "g_l1" in h:
            ax[1].plot(ep, h["g_l1"], label="G aligned-L1", color="#a371f7")
        if h.get("g_lm"):
            ax[1].plot(ep, h["g_lm"], label="G landmark", color="#3fb950")
        ax[1].set_title(f"Model {k}: generator terms"); ax[1].set_xlabel("epoch")
        ax[1].legend(); ax[1].grid(alpha=.3)
        plt.suptitle(title, y=1.02, fontsize=13)
        plt.tight_layout()
        f = os.path.join(OUT, f"loss_model_{k}.png")
        plt.savefig(f, dpi=150, bbox_inches="tight"); plt.close()
        print("wrote", f)


def chart_d_loss_overlay():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False
    for k, (base, color, title) in MODELS.items():
        h = _load_hist(base)
        if not h or not h.get("d"):
            continue
        any_data = True
        ax.plot(range(1, len(h["d"]) + 1), h["d"], label=f"Model {k}", color=color, lw=2)
    if not any_data:
        plt.close(); return
    ax.set_title("Discriminator loss across models"); ax.set_xlabel("epoch")
    ax.set_ylabel("D loss"); ax.legend(); ax.grid(alpha=.3)
    f = os.path.join(OUT, "d_loss_overlay.png")
    plt.savefig(f, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", f)


def chart_metric_bars():
    if not os.path.exists(RESULTS):
        print("no metrics.json yet - skipping metric bars"); return
    with open(RESULTS) as f:
        m = f and json.load(open(RESULTS))
    per = m.get("per_model", {})
    if not per:
        return
    ks = [k for k in ["A", "B", "C"] if k in per]
    colors = [MODELS[k][1] for k in ks]
    metrics = [("recognition", "GAN-test recognition ↑"),
               ("diversity", "Intra-class diversity ↑"),
               ("ssim", "SSIM to real ↑")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (mk, title) in zip(axes, metrics):
        vals = [per[k].get(mk) or 0 for k in ks]
        bars = ax.bar(ks, vals, color=colors)
        ax.set_title(title); ax.grid(alpha=.3, axis="y")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
    plt.suptitle("Model comparison (128px full run)", y=1.03, fontsize=13)
    plt.tight_layout()
    f = os.path.join(OUT, "metric_comparison.png")
    plt.savefig(f, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", f)


if __name__ == "__main__":
    chart_per_model_losses()
    chart_d_loss_overlay()
    chart_metric_bars()
    print("charts ->", OUT)
