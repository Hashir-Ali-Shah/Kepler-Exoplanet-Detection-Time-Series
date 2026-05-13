import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

DATA_PATH = "/kaggle/input/datasets/hashirali12123/exotrain/exoTrain.csv"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
RESULTS_DIR = "/kaggle/working/results"
EPOCHS = 50
BATCH_SIZE = 64
GAUSSIAN_SIGMA = 2
RANDOM_STATE = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading data...")
data = pd.read_csv(DATA_PATH)
X = data.drop("LABEL", axis=1)
y = data["LABEL"] - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("Applying Gaussian smoothing & row-wise scaling...")
X_train_sm = gaussian_filter1d(X_train, sigma=GAUSSIAN_SIGMA, axis=1)
X_test_sm = gaussian_filter1d(X_test, sigma=GAUSSIAN_SIGMA, axis=1)

def row_scale(arr):
    mu = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    return (arr - mu) / (std + 1e-8)

X_train_scaled = row_scale(X_train_sm)
X_test_scaled = row_scale(X_test_sm)
X_test_reshaped = np.expand_dims(X_test_scaled, axis=2)
y_test_arr = np.asarray(y_test)

TIMESTEPS = X_train_scaled.shape[1]
INPUT_SHAPE = (TIMESTEPS, 1)

def rotation_augment(X_scaled, y_arr, n_rotations=3):
    minority_idx = np.where(y_arr == 1)[0]
    X_min = X_scaled[minority_idx]
    X_rot, y_rot = [], []
    for _ in range(n_rotations):
        for sample in X_min:
            shift = np.random.randint(1, X_scaled.shape[1])
            X_rot.append(np.roll(sample, shift))
            y_rot.append(1)
    if X_rot:
        X_aug = np.vstack([X_scaled, np.array(X_rot)])
        y_aug = np.concatenate([y_arr, np.array(y_rot)])
    else:
        X_aug, y_aug = X_scaled, y_arr
    return X_aug, y_aug

def make_smote_set(X_scaled, y_arr):
    print("  Applying rotation augmentation + SMOTE...")
    X_aug, y_aug = rotation_augment(X_scaled, y_arr, n_rotations=3)
    k = min(5, int(y_aug[y_aug == 1].sum()) - 1)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_aug, y_aug)
    print(f"  SMOTE distribution: {np.bincount(y_res)}")
    return np.expand_dims(X_res, axis=2), y_res

def windowed_oversample(X_scaled, y_arr, window_size=200, step=50):
    print("  Applying windowed oversampling...")
    minority_idx = np.where(y_arr == 1)[0]
    T = X_scaled.shape[1]
    X_new, y_new = [], []
    for idx in minority_idx:
        sample = X_scaled[idx]
        start = 0
        while start + window_size <= T:
            window = sample[start:start+window_size]
            x_old = np.linspace(0, 1, window_size)
            x_new = np.linspace(0, 1, T)
            upsampled = np.interp(x_new, x_old, window)
            X_new.append(upsampled)
            y_new.append(1)
            start += step
    if X_new:
        X_aug = np.vstack([X_scaled, np.array(X_new)])
        y_aug = np.concatenate([y_arr, np.array(y_new)])
    else:
        X_aug, y_aug = X_scaled, y_arr
    print(f"  Windowed oversample distribution: {np.bincount(y_aug)}")
    return np.expand_dims(X_aug, axis=2), y_aug

y_train_arr = np.asarray(y_train)
print("Building SMOTE training set...")
X_train_smote, y_train_smote = make_smote_set(X_train_scaled, y_train_arr)
print("Building windowed-oversampled training set...")
X_train_win, y_train_win = windowed_oversample(X_train_scaled, y_train_arr)

focal_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

def build_lstm(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(32, return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dense(1, activation="sigmoid"),
    ])
    return model

def build_gru(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),
        GRU(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),
        GRU(32, return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),
        GRU(32, return_sequences=False),
        BatchNormalization(),
        Dense(1, activation="sigmoid"),
    ])
    return model

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=focal_loss,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

def get_initial_epoch(ckpt_dir):
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("last_epoch", 0)
    return 0

def save_meta(ckpt_dir, epoch):
    meta_path = os.path.join(ckpt_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({"last_epoch": epoch}, f)

def load_or_build(model_name, builder, ckpt_dir):
    weights_path = os.path.join(ckpt_dir, "weights.weights.h5")
    model = builder(INPUT_SHAPE)
    compile_model(model)
    if os.path.exists(weights_path):
        print(f"  [{model_name}] Resuming from checkpoint...")
        model.load_weights(weights_path)
    else:
        print(f"  [{model_name}] Starting from scratch...")
    return model

def train_model(model_name, model, X_tr, y_tr, ckpt_dir):
    initial_epoch = get_initial_epoch(ckpt_dir)
    if initial_epoch >= EPOCHS:
        print(f"  [{model_name}] Already fully trained ({EPOCHS} epochs). Skipping.")
        return model
    weights_path = os.path.join(ckpt_dir, "weights.weights.h5")
    best_weights_path = os.path.join(ckpt_dir, "best_weights.weights.h5")
    
    class EpochCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.model.save_weights(weights_path)
            save_meta(ckpt_dir, epoch + 1)
            
    class MinorityMetrics(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            y_prob = self.model.predict(X_test_reshaped, verbose=0).ravel()
            y_pred = (y_prob > 0.5).astype(int)
            p = precision_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
            r = recall_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
            f = f1_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
            print(f"  [Epoch {epoch+1}] Class-1 -> Precision: {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}")
            
    best_val_auc = [0.0]
    class SaveBestWeights(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            current = logs.get("val_auc", 0.0)
            if current > best_val_auc[0]:
                best_val_auc[0] = current
                self.model.save_weights(best_weights_path)
                
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", factor=0.5, patience=5, min_lr=1e-6, mode="max", verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=15, restore_best_weights=True, mode="max", verbose=1
    )
    
    model.fit(
        X_tr, y_tr,
        validation_data=(X_test_reshaped, y_test_arr),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        initial_epoch=initial_epoch,
        callbacks=[EpochCheckpoint(), MinorityMetrics(), SaveBestWeights(), reduce_lr, early_stop],
        verbose=1,
    )
    if os.path.exists(best_weights_path):
        print(f"  [{model_name}] Loading best weights (val_auc={best_val_auc[0]:.4f})...")
        model.load_weights(best_weights_path)
    return model

def evaluate_model(model_name, model):
    print(f"\n{'='*60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*60}")
    y_prob = model.predict(X_test_reshaped, verbose=0).ravel()
    print("\n--- Threshold sweep ---")
    for t in np.linspace(0.1, 0.9, 9):
        y_pred = (y_prob > t).astype(int)
        p1 = precision_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
        r1 = recall_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
        p0 = precision_score(y_test_arr, y_pred, pos_label=0, zero_division=0)
        r0 = recall_score(y_test_arr, y_pred, pos_label=0, zero_division=0)
        f0 = f1_score(y_test_arr, y_pred, pos_label=0, zero_division=0)
        print(f"  Threshold {t:.1f} | Class1 P={p1:.3f} R={r1:.3f} F1={f1:.3f} | Class0 P={p0:.3f} R={r0:.3f} F1={f0:.3f}")
    
    y_pred_05 = (y_prob > 0.5).astype(int)
    print("\n--- Classification report @ threshold=0.5 ---")
    print(classification_report(y_test_arr, y_pred_05, digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_arr, y_pred_05))
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test_arr, y_prob)
    pr_auc = average_precision_score(y_test_arr, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_vals, precision_vals, label=f"AP = {pr_auc:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {model_name}")
    ax.legend(); ax.grid()
    pr_path = os.path.join(RESULTS_DIR, f"{model_name}_pr_curve.png")
    fig.savefig(pr_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  PR Curve saved -> {pr_path}  (PR-AUC = {pr_auc:.4f})")
    
    fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(); ax.grid()
    roc_path = os.path.join(RESULTS_DIR, f"{model_name}_roc_curve.png")
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC Curve saved -> {roc_path}  (ROC-AUC = {roc_auc:.4f})")

MODELS = [
    {"name": "LSTM_SMOTE", "builder": build_lstm, "X_tr": X_train_smote, "y_tr": y_train_smote, "ckpt": os.path.join(CHECKPOINT_DIR, "lstm_smote")},
    {"name": "GRU_SMOTE", "builder": build_gru, "X_tr": X_train_smote, "y_tr": y_train_smote, "ckpt": os.path.join(CHECKPOINT_DIR, "gru_smote")},
    {"name": "LSTM_Windowed", "builder": build_lstm, "X_tr": X_train_win, "y_tr": y_train_win, "ckpt": os.path.join(CHECKPOINT_DIR, "lstm_windowed")},
    {"name": "GRU_Windowed", "builder": build_gru, "X_tr": X_train_win, "y_tr": y_train_win, "ckpt": os.path.join(CHECKPOINT_DIR, "gru_windowed")},
]

for cfg in MODELS:
    os.makedirs(cfg["ckpt"], exist_ok=True)
    print(f"\n{'#'*60}")
    print(f"  Model: {cfg['name']}")
    print(f"{'#'*60}")
    model = load_or_build(cfg["name"], cfg["builder"], cfg["ckpt"])
    model = train_model(cfg["name"], model, cfg["X_tr"], cfg["y_tr"], cfg["ckpt"])
    evaluate_model(cfg["name"], model)
print("\nAll models done. Results saved in:", RESULTS_DIR)