import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from softmax_regression import SoftmaxRegression

# Chia tập theo tỉ lệ của các nhãn
def train_test_split(X, y_indices, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    unique_classes = np.unique(y_indices)
    train_indices = np.array([], dtype=int)
    test_indices = np.array([], dtype=int)
    
    for c in unique_classes:
        c_indices = np.where(y_indices == c)[0]
        np.random.shuffle(c_indices)
        
        n_test = int(len(c_indices) * test_size)
        
        test_indices = np.concatenate((test_indices, c_indices[:n_test]))
        train_indices = np.concatenate((train_indices, c_indices[n_test:]))
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return X[train_indices], X[test_indices], train_indices, test_indices

df = pd.read_csv(Path(__file__).resolve().parents[2] / "LAB1" / "BAI2" / "data" / "yugioh.csv")

features = ['atk', 'def', 'level_rank', 'linkval', 'scale', 'link_marker_count']

X = df[features].fillna(0).values
y_raw = df['card_category'].values

unique_classes = np.unique(y_raw)
K = len(unique_classes)
class_to_idx = {val: idx for idx, val in enumerate(unique_classes)}
y_indices = np.array([class_to_idx[val] for val in y_raw])

y_one_hot = np.eye(K)[y_indices]

# Chuẩn hóa dữ liệu
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0
X = (X - X_mean) / X_std

# Chia tập Train / Test
train_data, test_data, train_idx, test_idx = train_test_split(
    X, y_indices, test_size=0.2, random_state=42
)

train_labels_one_hot = y_one_hot[train_idx]
test_labels_one_hot = y_one_hot[test_idx]

test_labels = test_labels_one_hot

# Huấn luyện
model = SoftmaxRegression(epochs=1000, learning_rate=0.1)
model.fit(train_data, train_labels_one_hot)

print(train_labels_one_hot.shape)

# Đánh giá
metrics = model.evaluate(test_data, test_labels)

print("Kết quả Bài toán 2 (Softmax Regression - NumPy):")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Visualize loss function
plt.figure(figsize=(8, 5))
plt.plot(model.loss_history, color='blue', linewidth=2)
plt.title('Training Loss Function - Softmax Regression')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('softmax_regression_numpy_loss_function.png', dpi=300, bbox_inches='tight')
plt.show()