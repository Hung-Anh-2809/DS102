import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class SoftmaxRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        self.w = None
        self.b = None
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_history = []
        
    def loss_fn(self, y_hat: np.ndarray, y: np.ndarray):
        """Tính toán giá trị hàm mất mát Cross-Entropy đa lớp

        Args:
            y_hat (np.ndarray): (N, K)
            y (np.ndarray): (N, K)

        Returns:
            float: Giá trị mất mát trung bình trên toàn bộ tập dữ liệu.
        """
        
        return - (y * np.log(y_hat + 1e-15)).sum(axis=1).mean()
        
    def softmax(self, X: np.ndarray):
        """Tính toán logits và kích hoạt bằng hàm Softmax

        Args:
            X (np.ndarray): (N, D)

        Returns:
            np.ndarray: (N, K)
        """
        
        logits = X @ self.w + self.b
        logits = logits - np.max(logits, axis=1, keepdims=True)
        logits = np.exp(logits)
        
        return logits / np.sum(logits, axis=1, keepdims=True)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện trọng số w và độ chệch b bằng thuật toán Gradient Descent

        Args:
            X (np.ndarray): (N, D)
            y (np.ndarray): (N, K)
        """
        
        N, D = X.shape
        K = y.shape[1]
        
        self.w = np.zeros((D, K))
        self.b = np.zeros((K, ))
        
        for _ in range(self.epochs):
            y_hat = self.softmax(X)
            
            loss = self.loss_fn(y_hat, y)
            self.loss_history.append(loss)
            
            grad_w = X.T @ (y_hat - y)
            grad_b = np.sum(y_hat - y, axis=0)
            
            self.w = self.w - (1 / N) * grad_w * self.lr
            self.b = self.b - (1 / N) * grad_b * self.lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán đa nhãn cho dữ liệu đầu vào

        Args:
            X (np.ndarray): (N, D)

        Returns:
            np.ndarray: (N, )
        """
        
        logits = self.softmax(X)
        predicted_y = np.argmax(logits, axis=-1)
        
        return predicted_y
    
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """Đánh giá hiệu năng mô hình bằng Precision, Recall và F1-Score

        Args:
            X (np.ndarray): (N, D)
            y (np.ndarray): (N, ) hoặc (N, K)

        Returns:
            dict: Điểm của ba chỉ số
        """
        
        y_hat = self.predict(X)

        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=-1)
            
        P = precision_score(y, y_hat, average="macro")
        R = recall_score(y, y_hat, average="macro")
        f1 = f1_score(y, y_hat, average="macro")
        
        return {
            "Precision": P,
            "Recall": R,
            "F1 Score": f1
        }
