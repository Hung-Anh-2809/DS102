import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from logistic_regression import LogisticRegression

df = pd.read_csv(Path(__file__).resolve().parents[2] / "LAB1" / "BAI2" / "data" / "yugioh.csv")

# Chia tập theo tỉ lệ của các nhãn
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    unique_classes = np.unique(y)
    train_indices = np.array([], dtype=int)
    test_indices = np.array([], dtype=int)
    
    for c in unique_classes:
        c_indices = np.where(y == c)[0]
        np.random.shuffle(c_indices)
        
        n_test = int(len(c_indices) * test_size)
        
        test_indices = np.concatenate((test_indices, c_indices[:n_test]))
        train_indices = np.concatenate((train_indices, c_indices[n_test:]))
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

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
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện
model = LogisticRegression(epochs=1000, learning_rate=0.1)
model.fit(train_data, train_labels)

# Đánh giá
metrics = model.evaluate(test_data, test_labels)

print("Kết quả Bài toán 1 (Logistic Regression - NumPy):")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Visualize loss function
plt.figure(figsize=(8, 5))
plt.plot(model.loss_history, color='blue', linewidth=2)
plt.title('Training Loss Function - Logistic Regression')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('logistic_regression_numpy_loss_function.png', dpi=300, bbox_inches='tight')
plt.show()