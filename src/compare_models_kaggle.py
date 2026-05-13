# =============================================================================
# Model Comparison Script — KAGGLE VERSION
# Run this as a new cell after training + save_models.py are done.
# Loads all 4 saved models, compares them visually, zips all plots.
# =============================================================================

import os, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score,
    roc_curve, auc
)
from IPython.display import display, FileLink

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH      = "/kaggle/input/datasets/hashirali12123/exotrain/exoTrain.csv"
RESULTS_DIR    = "/kaggle/working/results"       # where .keras files are saved
COMPARE_DIR    = "/kaggle/working/comparison"    # output folder for plots
GAUSSIAN_SIGMA = 2
RANDOM_STATE   = 42
THRESHOLDS     = np.round(np.linspace(0.1, 0.9, 9), 2)

os.makedirs(COMPARE_DIR, exist_ok=True)

MODEL_PATHS = {
    "LSTM + SMOTE":    os.path.join(RESULTS_DIR, "LSTM_SMOTE.keras"),
    "GRU + SMOTE":     os.path.join(RESULTS_DIR, "GRU_SMOTE.keras"),
    "LSTM + Windowed": os.path.join(RESULTS_DIR, "LSTM_Windowed.keras"),
    "GRU + Windowed":  os.path.join(RESULTS_DIR, "GRU_Windowed.keras"),
}

COLORS = {
    "LSTM + SMOTE":    "#4C72B0",
    "GRU + SMOTE":     "#DD8452",
    "LSTM + Windowed": "#55A868",
    "GRU + Windowed":  "#C44E52",
}

# =============================================================================
# STEP 1 — REBUILD TEST SET (same pipeline as training)
# =============================================================================
print("Loading and preprocessing data...")
data = pd.read_csv(DATA_PATH)

X = data.drop("LABEL", axis=1)
y = data["LABEL"] - 1

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

X_test_sm     = gaussian_filter1d(X_test, sigma=GAUSSIAN_SIGMA, axis=1)
test_mean     = X_test_sm.mean(axis=1, keepdims=True)
test_std      = X_test_sm.std(axis=1,  keepdims=True)
X_test_scaled = (X_test_sm - test_mean) / (test_std + 1e-8)
X_test_3d     = np.expand_dims(X_test_scaled, axis=2)
y_true        = np.asarray(y_test)

print(f"Test set: {X_test_3d.shape}  |  Exoplanets (Class 1): {y_true.sum()}")

# =============================================================================
# STEP 2 — LOAD MODELS & COMPUTE METRICS
# =============================================================================
focal_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

probs   = {}
metrics = {}

for name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        print(f"  ⚠️  {name}: not found at {path} — skipping.")
        continue
    print(f"  Loading {name}...")
    model = tf.keras.models.load_model(path, compile=False)
    model.compile(optimizer="adam", loss=focal_loss,
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    y_prob = model.predict(X_test_3d, verbose=0).ravel()
    probs[name] = y_prob

    thresh_data = {}
    for t in THRESHOLDS:
        y_pred = (y_prob > t).astype(int)
        thresh_data[t] = {
            "p1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "r1": recall_score(   y_true, y_pred, pos_label=1, zero_division=0),
            "f1": f1_score(       y_true, y_pred, pos_label=1, zero_division=0),
            "p0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            "r0": recall_score(   y_true, y_pred, pos_label=0, zero_division=0),
            "f0": f1_score(       y_true, y_pred, pos_label=0, zero_division=0),
        }
    metrics[name] = thresh_data

models_loaded = list(probs.keys())
print(f"\nModels ready: {models_loaded}")

# =============================================================================
# PLOT 1 — Overlaid ROC Curves
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
for name in models_loaded:
    fpr, tpr, _ = roc_curve(y_true, probs[name])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=COLORS[name], lw=2,
            label=f"{name}  (AUC={roc_auc:.4f})")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=10);  ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(COMPARE_DIR, "01_compare_roc.png"), dpi=150)
plt.close(fig)
print("Saved: 01_compare_roc.png")

# =============================================================================
# PLOT 2 — Overlaid PR Curves
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
for name in models_loaded:
    prec, rec, _ = precision_recall_curve(y_true, probs[name])
    ap = average_precision_score(y_true, probs[name])
    ax.plot(rec, prec, color=COLORS[name], lw=2,
            label=f"{name}  (AP={ap:.4f})")
baseline = y_true.sum() / len(y_true)
ax.axhline(baseline, color="gray", ls="--", alpha=0.5,
           label=f"Baseline (no-skill = {baseline:.3f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=10);  ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(COMPARE_DIR, "02_compare_pr.png"), dpi=150)
plt.close(fig)
print("Saved: 02_compare_pr.png")

# =============================================================================
# PLOT 3 — Class-1 Precision / Recall / F1 vs Threshold (line plots)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, key, title in zip(
    axes,
    ["p1", "r1", "f1"],
    ["Precision (Class 1 — Exoplanet)",
     "Recall (Class 1 — Exoplanet)",
     "F1 Score (Class 1 — Exoplanet)"]
):
    for name in models_loaded:
        vals = [metrics[name][t][key] for t in THRESHOLDS]
        ax.plot(THRESHOLDS, vals, marker="o", color=COLORS[name], lw=2, label=name)
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(THRESHOLDS)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9);  ax.grid(alpha=0.3)

fig.suptitle("Class-1 Metrics vs Threshold — All Models",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(COMPARE_DIR, "03_metrics_vs_threshold.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 03_metrics_vs_threshold.png")

# =============================================================================
# PLOT 4 — Grouped Bar Chart at Key Thresholds
# =============================================================================
key_thresholds = [0.1, 0.2, 0.3, 0.5]
bar_keys   = ["p1", "r1", "f1"]
bar_labels = ["Precision", "Recall", "F1"]
x     = np.arange(len(bar_keys))
width = 0.18

fig, axes = plt.subplots(1, len(key_thresholds), figsize=(20, 5), sharey=True)
for ax, t in zip(axes, key_thresholds):
    for i, name in enumerate(models_loaded):
        vals = [metrics[name][t][k] for k in bar_keys]
        bars = ax.bar(x + i * width, vals, width,
                      label=name, color=COLORS[name], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title(f"Threshold = {t:.1f}", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * (len(models_loaded) - 1) / 2)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Score (Class 1 — Exoplanet)", fontsize=11)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(models_loaded),
           fontsize=10, bbox_to_anchor=(0.5, 1.05))
fig.suptitle("Class-1 Metrics at Key Thresholds", fontsize=14,
             fontweight="bold", y=1.10)
fig.tight_layout()
fig.savefig(os.path.join(COMPARE_DIR, "04_bar_key_thresholds.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 04_bar_key_thresholds.png")

# =============================================================================
# PLOT 5/6/7 — Heatmaps (Recall, Precision, F1)
# =============================================================================
for plot_num, (metric_key, metric_label) in enumerate(
    [("r1", "Recall"), ("p1", "Precision"), ("f1", "F1")], start=5
):
    hm_data = {
        name: [metrics[name][t][metric_key] for t in THRESHOLDS]
        for name in models_loaded
    }
    df_hm = pd.DataFrame(hm_data, index=[f"t={t:.1f}" for t in THRESHOLDS])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_hm, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size": 10})
    ax.set_title(f"Class-1 {metric_label} — Models × Thresholds",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Threshold", fontsize=11)
    fig.tight_layout()
    fname = f"0{plot_num}_heatmap_{metric_label.lower()}.png"
    fig.savefig(os.path.join(COMPARE_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

# =============================================================================
# ZIP ALL PLOTS + DOWNLOAD LINK
# =============================================================================
ZIP_PATH = "/kaggle/working/comparison_results.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(COMPARE_DIR)):
        fpath = os.path.join(COMPARE_DIR, fname)
        zf.write(fpath, arcname=fname)
        print(f"  Zipped: {fname}")

print(f"\nAll done! {len(os.listdir(COMPARE_DIR))} plots saved.")
print(f"Zip → {ZIP_PATH}")
display(FileLink("comparison_results.zip"))
