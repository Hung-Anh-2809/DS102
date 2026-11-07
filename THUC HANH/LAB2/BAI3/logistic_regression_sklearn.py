import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv(Path(__file__).resolve().parents[2] / "LAB1" / "BAI2" / "data" / "yugioh.csv")

# Chỉ lọc các dòng là Quái thú (card_category == 0)
df_monster = df[df['card_category'] == 0].copy()

features = ['atk', 'def', 'level_rank', 'linkval', 'scale', 'link_marker_count']

X = df_monster[features].fillna(0).values

# Phân loại có phải là Extra Deck hay không
y = df_monster['is_extra_deck'].values

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
P = precision_score(test_labels, y_predicted)
R = recall_score(test_labels, y_predicted)
f1 = f1_score(test_labels, y_predicted)

print("Kết quả Bài toán 1 (Logistic Regression - Scikit Learn):")
print(f"Precision: {P:.4f}")
print(f"Recall: {R:.4f}")
print(f"F1 Score: {f1:.4f}")