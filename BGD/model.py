import numpy as np
import time

class BGD:
    def __init__(self, batch_size = 30, learning_rate = 0.001):
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.data_len = X.shape[0]
        self.features_len = X.shape[1]
        
        self.batches = self.get_batches()
        self.w = np.random.randn(self.features_len)
        self.bias = 0

    def get_batches(self):
        i = 0
        batches = []
        while i < self.data_len:
            batches.append([
                self.X[i : i + self.batch_size],
                self.y[i : i + self.batch_size]
            ])
            i += self.batch_size
        return batches

    def h(self, x):
        return x.dot(self.w) + self.bias
    
    def J(self, x, y):
        return np.square((self.h(x) - y)).sum()/ x.shape[0] / 2
    
    def dJ(self, x, y):
        return (((self.h(x) - y) * x.T)).sum(axis=1) / x.shape[0]
    
    def bias_dJ(self, x, y):
        return ((self.h(x) - y)).sum() / x.shape[0] 
    
    def print_loss(self, x, y):
        print("\r", end="")
        print(f"loss: {self.J(x, y)}", end="")
        # time.sleep(0.01)
    
    def train(self, iterations = 100):
        for _ in range(iterations):
            for batch in self.batches:
                x_train, y_train = batch
                self.w -= self.learning_rate * self.dJ(x_train, y_train)
                self.bias -= self.learning_rate * self.bias_dJ(x_train, y_train)
                self.print_loss(x_train, y_train)
        print()

    def forward(self, x):
        return self.h(x)
