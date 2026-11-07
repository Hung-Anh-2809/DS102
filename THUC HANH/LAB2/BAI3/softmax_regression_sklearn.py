import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv(Path(__file__).resolve().parents[2] / "LAB1" / "BAI2" / "data" / "yugioh.csv")

features = ['atk', 'def', 'level_rank', 'linkval', 'scale', 'link_marker_count']

X = df[features].fillna(0).values
y = df['card_category'].values

# Chuẩn hóa dữ liệu
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0
X = (X - X_mean) / X_std

# Chia tập Train / Test
train_data, test_data, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Huấn luyện
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(train_data, train_labels)

# Dự đoán
y_predicted = model.predict(test_data)

# Đánh giá
P = precision_score(test_labels, y_predicted, average='macro')
R = recall_score(test_labels, y_predicted, average='macro')
f1 = f1_score(test_labels, y_predicted, average='macro')

print("Kết quả Bài toán 2 (Softmax Regression - Scikit Learn):")
print(f"Precision: {P:.4f}")
print(f"Recall: {R:.4f}")
print(f"F1 Score: {f1:.4f}")