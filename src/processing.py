import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d

data = pd.read_csv("exoTrain.csv")
scaler = StandardScaler()
X = data.drop('LABEL', axis=1)
y = data['LABEL']
y = y - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_smoothed = gaussian_filter1d(X_train, sigma=3, axis=1)
X_test_smoothed = gaussian_filter1d(X_test, sigma=3, axis=1)

train_mean = X_train_smoothed.mean(axis=1).reshape(-1, 1)
train_std = X_train_smoothed.std(axis=1).reshape(-1, 1)
X_train_scaled = (X_train_smoothed - train_mean) / (train_std + 1e-8)

test_mean = X_test_smoothed.mean(axis=1).reshape(-1, 1)
test_std = X_test_smoothed.std(axis=1).reshape(-1, 1)
X_test_scaled = (X_test_smoothed - test_mean) / (test_std + 1e-8)

X_train_reshaped = np.expand_dims(X_train_scaled, axis=2)
X_test_reshaped = np.expand_dims(X_test_scaled, axis=2)