import numpy as np

class LogRegression:
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.size = X.shape[0]
        self.n_features = X.shape[1]
        self.w = np.random.randn(self.n_features + 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, x):
        return self.sigmoid(self.w @ x)
    
    def log_loss(self, x, y):
        p = self.predict(x)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)
    
    def der(self, x, y):
        return (self.predict(x) - y) * x

    def train(self):
        for e in range(1000):
            for i in range(self.size):
                x = np.concatenate([[1.0], self.X[i]])
                self.w -= self.der(x, self.y[i])