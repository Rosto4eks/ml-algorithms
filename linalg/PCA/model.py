import numpy as np

# principal component analysis
class PCA:
    def __init__(self, rate = 0.8):
        self.rate = rate

    def fit(self, X):
        X = np.array(X)

        u, s, v = np.linalg.svd(X)
        max_loss = 1 - self.rate
        loss = 0
        new_len = v.shape[0]
        s = s / np.sum(s)
        for e in reversed(s):
            if loss + e > max_loss:
                break
            loss += e
            new_len -= 1
        vecs = v.T[0:new_len].T

        return X.dot(vecs)