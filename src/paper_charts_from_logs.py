"""
Loss charts for Models F and G rebuilt from the committed training logs.

progress.json / loss_history.csv are gitignored and live only on the training
machine, so the per-epoch values are parsed out of reports/paper/logs/train_*.log
into reports/paper/results/history_from_logs/loss_history_{B,C,F,G}.csv. This
script plots F and G from those CSVs with the adversarial and landmark terms on
separate panels (they differ by ~3 orders of magnitude).

Run:  python src/paper_charts_from_logs.py
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(REPO, "reports", "paper", "results", "history_from_logs")
OUT = os.path.join(REPO, "reports", "paper", "charts")

MODELS = {
    "F": ("#58a6ff", "F — structure-conditioned + landmark loss"),
    "G": ("#f778ba", "G — F + aux-classifier + feature matching + EMA"),
}


def _load(k):
    with open(os.path.join(HIST, f"loss_history_{k}.csv")) as f:
        rows = list(csv.DictReader(f))
    return {c: [float(r[c]) for r in rows] for c in rows[0]}


def main():
    for k, (color, title) in MODELS.items():
        h = _load(k)
        ep = h["epoch"]
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
        ax[0].plot(ep, h["d"], label="D loss", color="#f85149")
        ax[0].plot(ep, h["g"], label="G total", color=color)
        ax[0].set_title(f"Model {k}: D and total G loss")
        ax[1].plot(ep, h["g_adv"], label="G adversarial", color="#58a6ff")
        ax[1].plot(ep, h["g_l1"], label="G aligned-L1", color="#a371f7")
        if "g_fm" in h:
            ax[1].plot(ep, h["g_fm"], label="G feature-matching", color="#d29922")
        ax[1].set_title(f"Model {k}: adversarial and reconstruction terms")
        ax[2].plot(ep, h["g_lm"], label="landmark loss (unweighted)", color="#3fb950")
        ax[2].plot(ep, [l * w for l, w in zip(h["g_lm"], h["lam_lm"])],
                   label="λ_lm × landmark (as added to G loss)", color="#3fb950", ls="--")
        ax[2].set_title(f"Model {k}: landmark term")
        for a in ax:
            a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
        plt.suptitle(title, y=1.02, fontsize=13)
        plt.tight_layout()
        f = os.path.join(OUT, f"loss_model_{k}.png")
        plt.savefig(f, dpi=150, bbox_inches="tight"); plt.close()
        print("wrote", f)


if __name__ == "__main__":
    main()
