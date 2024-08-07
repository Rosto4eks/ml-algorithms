import numpy as np
from sklearn.model_selection import train_test_split

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
    
    def copy(self):
        return Node(self.f_index, self.f_val, self.y)

    
class Cart:
    def __init__(self, type = "c", pruning = True, alpha = 0.001, max_height = np.inf, node_min_size = 1, min_allowed_Gini = 0):
        self.type = type
        self.pruning = pruning
        self.alpha = alpha
        self.max_height = max_height
        self.node_min_size = node_min_size
        self.min_allowed_Gini = min_allowed_Gini


    def fit(self, X, y):
        if self.pruning:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
            self.root = self.branch(X_train, y_train)
            self.root = self.prune(self.root, X_val, y_val)
        else:
            self.root = self.branch(X, y)

    def branch(self, X, y, height = 0):
        n = Node()
        if self.stop_criteria(y, height):
            n.y = self.get_leaf_val(y)
            return n
        
        f_index, f_val = self.fitNode(X, y)
        n.f_index = f_index
        n.f_val = f_val

        mask = n.predict(X)

        n.left = self.branch(X[mask], y[mask], height + 1)
        n.right = self.branch(X[~mask], y[~mask], height + 1)
        return n
    
    def fitNode(self, X, y):
        inf_dif = np.inf
        inf_index = None
        inf_val = None

        for f_index in range(X.shape[1]):
            indices = np.argsort(X[:, f_index])
            xs = X[indices]
            ys = y[indices]
            for p_index in range(1, xs.shape[0]):
                pred = (xs[p_index][f_index] + xs[p_index - 1][f_index]) / 2
                mask = xs[:, f_index] < pred

                left_H = self.H(ys[mask])
                right_H = self.H(ys[~mask])
                left_c = len(ys[mask]) / len(ys)
                inf = left_c * left_H + (1 - left_c) * right_H

                if inf < inf_dif:
                    inf_dif = inf
                    inf_index = f_index
                    inf_val = pred
            
        return inf_index, inf_val
    
    def prune(self, node, X, y):
        if node.y != None:
            return node

        mask = node.predict(X)
        if mask.sum() == len(mask) or mask.sum() == 0:
            return Node(y=self.get_leaf_val(y))
            
        node.left = self.prune(node.left, X[mask], y[mask])
        node.right = self.prune(node.right, X[~mask], y[~mask])

        nodes =[Node(y=self.get_leaf_val(y)), node, node.left, node.right]
        preds = [self.predictNode(X, n) for n in nodes]
        errs = [self.MSE(preds[i], y, nodes[i]) for i in range(4)]

        return nodes[np.argmin(errs)]

     
    def stop_criteria(self, y, height):
        c1 = height >= self.max_height
        c2 = self.Gini(y) <= self.min_allowed_Gini
        c3 = len(y) <= self.node_min_size
        return c1 | c2 | c3
    
    def get_leaf_val(self, Y):
        if self.type == "c":
            vals, counts = np.unique(Y, return_counts=True)
            return vals[np.argmax(counts)]
        else:
            return np.mean(Y)


    def H(self, Y):
        return self.Gini(Y) if self.type == "c" else np.var(Y)

    def Gini(self, Y):
        _ , counts = np.unique(Y, return_counts=True)
        probs = counts / len(Y)
        return sum(p * (1 - p) for p in probs)
    
    def MSE(self, Y, y, node):
        return np.square(Y - y).sum() / len(Y) + self.alpha * self.size(node)
    
    def predict(self, X):
        return self.predictNode(X, self.root)
    
    def predictNode(self, X, node):
        if np.ndim(X) == 1:
            X = X[np.newaxis, :]

        predictions = np.empty(shape=X.shape[0])
        for i in range(len(X)):
            n = node
            while n.y == None:
                n = n.left if n.predict(X[i]) else n.right 

            predictions[i] = n.y

        return predictions
    
    def size(self, node):
        if node.y != None:
            return 1
        return 1 + self.size(node.left) + self.size(node.right)
