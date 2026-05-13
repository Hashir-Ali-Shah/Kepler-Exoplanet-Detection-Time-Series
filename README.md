# Exoplanet Detection from Stellar Light Curves

This project implements and benchmarks deep learning models (LSTM and GRU) to detect exoplanets using time-series stellar flux data. The primary challenge is the extreme class imbalance (less than 1% positive samples) and high noise levels in the signal.

## 🚀 Key Features

*   **Architectures**: Benchmark comparing 2 LSTM and 2 GRU models.
*   **Imbalance Strategies**: Dual approach using **SMOTE** (Synthetic Minority Over-sampling Technique) and **Windowed Oversampling**.
*   **Signal Processing**: Gaussian smoothing ($\sigma=2$) and row-wise relative flux normalization.
*   **Robust Training**: 
    *   **Focal Loss**: To handle hard-to-classify samples.
    *   **Gradient Clipping**: Prevents catastrophic gradient explosions.
    *   **Fault Tolerance**: Per-epoch checkpointing to survive internet disconnections or session timeouts.
    *   **Dynamic Learning**: Automatic LR reduction on plateaus.

---

## 📊 Experimental Results

We evaluated 4 configurations on a hold-out test set (20% of data).

### Model Performance Summary

| Model | Architecture | Oversampling | ROC-AUC | PR-AUC | Best Recall (C1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1** | LSTM | SMOTE | 0.7712 | 0.0188 | 1.000 @ t=0.1 |
| **Model 2** | GRU | SMOTE | 0.7051 | 0.0135 | 1.000 @ t=0.1 |
| **Model 3** | LSTM | Windowed | 0.8139 | 0.0300 | 0.857 @ t=0.1 |
| **Model 4** | GRU | Windowed | **0.8970** | **0.0360** | **1.000 @ t=0.1** |

### Visual Comparisons

#### 1. ROC and Precision-Recall Curves
The ROC curves show that the **GRU + Windowed** model provides the best overall separation between classes. However, the Precision-Recall curves highlight the difficulty of achieving high precision in such a sparse dataset.

![ROC Curves](comparison_results/01_compare_roc.png)
![PR Curves](comparison_results/02_compare_pr.png)

#### 2. Metric Heatmaps (Recall vs Threshold)
To maximize the detection of rare exoplanets, we analyzed Recall across multiple probability thresholds. Lowering the threshold to $0.1$ ensures $100\%$ detection for most models, albeit at the cost of precision.

![Recall Heatmap](comparison_results/05_heatmap_recall.png)

---

## 🔬 Findings & Analysis

1.  **Windowed vs SMOTE**: Interestingly, Windowed Oversampling (extracting segments and upsampling) performed significantly better in terms of AUC than traditional SMOTE for this specific time-series task.
2.  **Training Stability**: Early versions of the training crashed due to gradient explosions. The addition of `clipnorm=1.0` and `ReduceLROnPlateau` stabilized the training significantly.
3.  **Thresholding**: Due to the imbalance, the default $0.5$ threshold is often too aggressive. Optimal deployment requires a sliding threshold (likely $0.1$ to $0.3$) depending on whether the priority is sensitivity (Recall) or precision.

---

## 🛠️ How to Run

1.  **Train**: Run `train_models.py` in a Kaggle environment. It will handle checkpoints automatically.
2.  **Export**: Run `save_models.py` to compile and save the `.keras` files.
3.  **Compare**: Run `compare_models_kaggle.py` to generate the visual report and download the zip.

---

## 📈 Future Work

While current results are promising, accuracy and precision for the minority class remain the primary focus for improvement. We plan to explore:
*   **Wavelet Transforms**: To extract better frequency-domain features from the light curves.
*   **Transformer Architectures**: Utilizing attention mechanisms to capture long-range dependencies better than standard RNNs.
*   **Advanced Augmentation**: Implementing jittering, scaling, and time-warping without distorting the transit signal.
*   **Ensemble Methods**: Combining the predictions of SMOTE and Windowed models to reduce false positives.

---
*Created as part of the Exoplanet Detection Research Project.*
