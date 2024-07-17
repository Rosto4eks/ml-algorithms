import numpy as np

# principal component analysis
class PCA:
    def fit(self, x):
        self.x = np.array(x)

    def eval(self, rate = 0.8):
        u, s, v = np.linalg.svd(self.x)
        max_loss = 1 - rate
        loss = 0
        new_len = v.shape[0]
        s = s / np.sum(s)
        for e in reversed(s):
            if loss + e > max_loss:
                break
            loss += e
            new_len -= 1
        vecs = v.T[0:new_len].T

        return self.x.dot(vecs)