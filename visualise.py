"""Plots: score distributions, PR curves, ROC curves."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_score_dist(score, y, thr, name, out="figures"):
    os.makedirs(out, exist_ok=True); y = np.asarray(y)
    plt.figure(figsize=(7, 4))
    lo, hi = np.percentile(score, [0.5, 99.9])
    bins = np.linspace(lo, hi, 80)
    plt.hist(score[y == 0], bins, alpha=.6, density=True, label="Legit")
    plt.hist(score[y == 1], bins, alpha=.6, density=True, label="Fraud")
    plt.axvline(thr, color="k", ls="--", label="Alert threshold")
    plt.xlabel("Anomaly score"); plt.title(f"{name}: score distribution")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{out}/score_dist.png", dpi=150); plt.close()


def plot_curves(scores, y, out="figures"):
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s); ax[0].plot(r, p, label=name)
        f, t, _ = roc_curve(y, s); ax[1].plot(f, t, label=name)
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax[1].set(xlabel="FPR", ylabel="TPR", title="ROC"); ax[1].plot([0, 1], [0, 1], "k--", alpha=.3)
    ax[0].legend(); plt.tight_layout(); plt.savefig(f"{out}/curves.png", dpi=150); plt.close()
