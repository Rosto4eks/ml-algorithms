import numpy as np
from tree import Cart

class GradientBoosting:
    def __init__(self, type="c", n=100, learning_rate=0.1, max_height=3):
        self.type = type
        self.n = n
        self.learning_rate = learning_rate
        self.max_height = max_height
        self.trees = []


    def fit(self, X, y):
        self.mean = np.mean(y)
        y_pred = np.full(y.shape, self.mean)

        for _ in range(self.n):
            y_next = y - y_pred
    
            tree = Cart(max_height=self.max_height, type="r")
            tree.fit(X, y_next)

            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)


    def predict(self, X):
        if np.ndim(X) == 1:
            X = X[np.newaxis, :]
        fm = np.full(X.shape[0], self.mean)
        for tree in self.trees:
            fm += self.learning_rate * tree.predict(X)

        if self.type == "c":
            fm = np.round(fm)
        return fm