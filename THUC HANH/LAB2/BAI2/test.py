# test_random_forest.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# 1. Đọc dữ liệu
df = pd.read_csv(Path(__file__).resolve().parents[2] / "LAB1" / "BAI2" / "data" / "yugioh.csv")

print("=== BÀI TOÁN 1: PHÂN LOẠI NHỊ PHÂN (Extra Deck vs Main Deck) ===")
df_monster = df[df['card_category'] == 0].copy()
features_bin = ['atk', 'def', 'level_rank', 'linkval', 'scale', 'link_marker_count']

X_bin = df_monster[features_bin].fillna(0).values
y_bin = df_monster['is_extra_deck'].values

# Chia tập Train/Test (80/20)
np.random.seed(42)
indices = np.random.permutation(len(X_bin))
split = int(0.8 * len(X_bin))
X_train, X_test = X_bin[indices[:split]], X_bin[indices[split:]]
y_train, y_test = y_bin[indices[:split]], y_bin[indices[split:]]

# Huấn luyện Random Forest
rf_bin = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_bin.fit(X_train, y_train)
y_pred_bin = rf_bin.predict(X_test)

print("Binary Precision:", precision_score(y_test, y_pred_bin, zero_division=0))
print("Binary Recall:   ", recall_score(y_test, y_pred_bin, zero_division=0))
print("Binary F1-Score: ", f1_score(y_test, y_pred_bin, zero_division=0))


print("\n=== BÀI TOÁN 2: PHÂN LOẠI ĐA LỚP (Card Category) ===")
features_multi = ['atk', 'def', 'level_rank', 'linkval', 'scale', 'link_marker_count', 'price_tcgplayer_usd']
X_multi = df[features_multi].fillna(0).values
y_multi = df['card_category'].values

indices_m = np.random.permutation(len(X_multi))
split_m = int(0.8 * len(X_multi))
X_train_m, X_test_m = X_multi[indices_m[:split_m]], X_multi[indices_m[split_m:]]
y_train_m, y_test_m = y_multi[indices_m[:split_m]], y_multi[indices_m[split_m:]]

# Huấn luyện Random Forest cho đa lớp
rf_multi = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_multi.fit(X_train_m, y_train_m)
y_pred_multi = rf_multi.predict(X_test_m)

print("Multi-class Macro Precision:", precision_score(y_test_m, y_pred_multi, average='macro', zero_division=0))
print("Multi-class Macro Recall:   ", recall_score(y_test_m, y_pred_multi, average='macro', zero_division=0))
print("Multi-class Macro F1-Score: ", f1_score(y_test_m, y_pred_multi, average='macro', zero_division=0))