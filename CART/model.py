import numpy as np

class Node:
    def __init__(self, f_index=None, f_val=None, y=None):
        self.f_index = f_index
        self.f_val = f_val
        self.y = y

        self.left = None
        self.right = None

    def predict(self, x):
        if np.ndim(x) == 1:
            x = x[np.newaxis, :]
        return x[:, self.f_index] < self.f_val

    

class Cart:
    def __init__(self, max_height = 3, node_min_size = 10, min_allowed_Gini = 0):
        self.max_height = max_height
        self.node_min_size = node_min_size
        self.min_allowed_Gini = min_allowed_Gini


    def fit(self, X, y):
        self.root = self.branch(X, y)

    def branch(self, X, y, height = 0):
        n = Node()
        if self.stop_criteria(y, height):
            vals, counts = np.unique(y, return_counts=True)
            n.y = vals[np.argmax(counts)]
            return n
        
        f_index, f_val = self.fitNode(X, y)
        n.f_index = f_index
        n.f_val = f_val

        mask = n.predict(X)

        n.left = self.branch(X[mask], y[mask], height + 1)
        n.right = self.branch(X[~mask], y[~mask], height + 1)
        return n
    
    def fitNode(self, X, y):
        inf_dif = -np.inf
        inf_index = None
        inf_val = None

        parent_gini = self.Gini(y)

        for f_index in range(X.shape[1]):
            indices = np.argsort(X[:, f_index])
            xs = X[indices]
            ys = y[indices]
            for p_index in range(1, xs.shape[0]):
                pred = (xs[p_index][f_index] + xs[p_index - 1][f_index]) / 2
                mask = xs[:, f_index] < pred

                left_gini = self.Gini(ys[mask])
                right_gini = self.Gini(ys[~mask])
                left_c = len(ys[mask]) / len(ys)
                inf = parent_gini - left_c * left_gini - (1 - left_c) * right_gini

                if inf > inf_dif:
                    inf_dif = inf
                    inf_index = f_index
                    inf_val = pred
            
        return inf_index, inf_val
                
    def stop_criteria(self, y, height):
        c1 = height >= self.max_height
        c2 = self.Gini(y) <= self.min_allowed_Gini
        c3 = len(y) <= self.node_min_size
        return c1 | c2 | c3


    def Gini(self, Y):
        _ , counts = np.unique(Y, return_counts=True)
        probs = counts / len(Y)
        return sum(p * (1 - p) for p in probs)
    
    def predict(self, x):
        n = self.root
        while n.y == None:
            n = n.left if n.predict(x) else n.right 

        return n.y