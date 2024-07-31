import numpy as np

class kNN:
    def __init__(self, k, n = 2, h = 1, n_trees = 10, max_leaf_elems = 10):
        self.k = k
        self.n = n
        self.h = h
        self.n_trees = n_trees
        self.max_leaf_elems = max_leaf_elems

    def dist(self, x_1, x_2):
        return np.linalg.norm(x_1 - x_2)

    def fit(self, X, y):
        forest = AnnoyForest(self.n_trees, self.max_leaf_elems)
        forest.fit(X, y)
        self.forest = forest

    def kernel(self, x):
        return np.exp(- 2 * x ** 2) / np.sqrt(2 * np.pi)

    def predict(self, x):
        xs, ys = self.forest.predict(x)

        marks = np.zeros(shape=self.n)
        for xi, yi in zip(xs, ys):
            marks[yi] += self.kernel(self.dist(x, xi) / self.h)
        return np.argmax(marks)
    

class AnnoyNode:
    def __init__(self, x_1, x_2):
        # build orthogonal hyperplane
        self.w = np.abs(x_1 - x_2)
        self.b = - np.sum(self.w * (x_1 + x_2)) / 2

        self.left = self.right = None

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)


class Annoy:
    def __init__(self, m = 10):
        self.m = m
        self.root = None

    def branch(self, X, y):
        if len(X) <= self.m:
            return X, y
        
        ids = np.random.randint(0, len(X), 2)
        node = AnnoyNode(X[ids[0]], X[ids[1]])
        mask = node.predict(X) < 0
        node.left = self.branch(X[mask], y[mask])
        node.right = self.branch(X[~mask], y[~mask])

        return node

    def fit(self, X, y):
        self.root = self.branch(X, y)

    def predict(self, x):
        node = self.root

        while isinstance(node, AnnoyNode):
            node = node.left if node.predict(x) < 0 else node.right
        return node
        
class AnnoyForest:  
    def __init__(self, n = 10, m = 10):
        self.n = n
        self.m = m

    def fit(self, X, y):
        self.forest = [Annoy(self.m) for _ in range(self.n)]
        for tree in self.forest:
            tree.fit(X, y)

    def predict(self, x):
        results = [tree.predict(x) for tree in self.forest]
        all_X = np.concatenate([X for X, _ in results])
        all_y = np.concatenate([y for _, y in results])
        unique_X, indices = np.unique(all_X, axis=0, return_index=True)
        unique_y = all_y[indices]
        return unique_X, unique_y


class sort_kNN:
    def __init__(self, k, n = 2, h = 1):
        self.k = k
        self.n = n
        self.h = h

    def dist(self, x_1, x_2):
        return np.linalg.norm(x_1 - x_2)

    def fit(self, X, y):
        self.X = X
        self.y = y

    def kernel(self, x):
        return np.exp(- 2 * x ** 2) / np.sqrt(2 * np.pi)

    def predict(self, x):
        arr = zip(self.X, self.y)
        
        sorted_arr = sorted(arr, key=lambda x_2: self.dist(x, x_2[0]))

        xs, ys = zip(*sorted_arr[:self.k])

        marks = np.zeros(shape=self.n)
        for i in range(len(xs)):
            marks[ys[i]] += self.kernel( self.dist(x, xs[i]) / self.h )
        return np.argmax(marks)