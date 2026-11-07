import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        self.w = None
        self.b = 0.0
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_history = []
        
    def loss_fn(self, y_hat: np.ndarray, y: np.ndarray):
        """Tính toán giá trị hàm mất mát Binary Cross-Entropy

        Args:
            y_hat (np.ndarray): (N, )
            y (np.ndarray): (N, )

        Returns:
            float: Giá trị mất mát trung bình của toàn bộ tập dữ liệu
        """
        
        return - (y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15)).mean()
        
    def sigmoid(self, X: np.ndarray):
        """Tính toán logits và kích hoạt bằng hàm Sigmoid

        Args:
            X (np.ndarray): (N, D)

        Returns:
            np.ndarray: (N, )
        """
        
        logits = X @ self.w + self.b
        logits = 1 / (1 + np.exp(-logits))
        
        return logits
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện trọng số w và độ chệch b bằng thuật toán Gradient Descent

        Args:
            X (np.ndarray): (N, D)
            y (np.ndarray): (N, )
        """
        
        N, D = X.shape
        self.w = np.zeros((D, ))
        self.b = 0.0
        
        for _ in range(self.epochs):
            y_hat = self.sigmoid(X)
            
            loss = self.loss_fn(y_hat, y)
            self.loss_history.append(loss)
            
            grad_w = X.T @ (y_hat - y)
            grad_b = np.sum(y_hat - y)
            
            self.w = self.w - (1 / N) * grad_w * self.lr
            self.b = self.b - (1 / N) * grad_b * self.lr

    def predict(self, x: np.ndarray):
        """Dự đoán nhãn nhị phân cho dữ liệu đầu vào

        Args:
            x (np.ndarray): (N, D)

        Returns:
            np.ndarray: (N, )
        """
        
        logits = self.sigmoid(x)
        predicted_y = np.where(logits >= 0.5, 1, 0)
        
        return predicted_y
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Đánh giá hiệu năng mô hình bằng Precision, Recall và F1-Score

        Args:
            X (np.ndarray): (N, D)
            y (np.ndarray): (N, )

        Returns:
            dict: 
        """
        
        y_hat = self.predict(X)
        
        P = precision_score(y, y_hat)
        R = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)
        
        return  {
            "Precision": P,
            "Recall": R,
            "F1 Score": f1
        }
