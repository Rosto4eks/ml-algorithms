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
    def __init__(self, type = "c", pruning = True, alpha = 0.0001, max_height = np.inf, node_min_size = 1, min_allowed_Gini = 0):
        self.type = type
        self.pruning = pruning
        self.alpha = alpha
        self.max_height = max_height
        self.node_min_size = node_min_size
        self.min_allowed_Gini = min_allowed_Gini


    def fit(self, X, y):
        if self.pruning:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            self.root = self.branch(self.X_train, self.y_train)
            self.root = self.prune(self.root, self.root.left)
            self.root = self.prune(self.root, self.root.right, False)
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
    
    def prune(self, parent, node, left = True):
        if node.y != None:
            return parent
        
        node = self.prune(node, node.left)
        node = self.prune(node, node.right, left = False)

        errs = [self.MSE(), 0, 0]
        nodes = [node, node.left, node.right]

        if left:
            parent.left = node.left
        else:
            parent.right = node.left
        errs[1] = self.MSE()

        if left:
            parent.left = node.right
        else:
            parent.right = node.right
        errs[2] = self.MSE()

        min_ind = np.argmin(errs)
        if left:
            parent.left = nodes[min_ind]
        else:
            parent.right = nodes[min_ind]
        
        return parent

     
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
    
    def MSE(self):
        return np.square(self.predict(self.X_val) - self.y_val).sum() / len(self.X_val) + self.alpha * self.size(self.root)
    
    
    def predict(self, X):
        if np.ndim(X) == 1:
            X = X[np.newaxis, :]

        predictions = np.empty(shape=X.shape[0])
        for i in range(len(X)):
            n = self.root
            while n.y == None:
                n = n.left if n.predict(X[i]) else n.right 

            predictions[i] = n.y

        return predictions
    
    def size(self, node):
        if node.y != None:
            return 1
        return 1 + self.size(node.left) + self.size(node.right)
