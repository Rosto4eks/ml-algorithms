import numpy as np
from tree import Cart
from scipy import stats
from multiprocessing import Pool

class RandomForest:
    def __init__(self, n=100, type="c"):
        self.n = n
        self.type = type
        self.trees = []


    def fit(self, X, y):
        self.X = X
        self.y = y
        with Pool(processes=8) as pool:
            trees = pool.map(self.fit_part, range(8))
        self.trees = np.ravel(trees)

    def fit_part(self, _):
        trees = []
        for _ in range(self.n // 8):
            ids = np.random.randint(0, self.X.shape[0], self.X.shape[0])
            X_i = self.X[ids]
            y_i = self.y[ids]
            tree = Cart(self.type, feature_list_size="auto")
            tree.fit(X_i, y_i)
            trees.append(tree)
        return trees
    

    def predict(self, X):
        if np.ndim(X) == 1:
            X = X[np.newaxis, :]

        if self.type == "c":
            preds = []
            for tree in self.trees:
                preds.append(tree.predict(X))

            return stats.mode(np.transpose(preds), axis=1).mode.flatten()
        else:
            preds = np.zeros(X.shape[0])
            for tree in self.trees:
                preds += tree.predict(X)
            
            return preds / self.n
