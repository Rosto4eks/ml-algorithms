import numpy as np

class Node:
    def __init__(self):
        self.index = -1
        self.mark = None
        self.feature = None
        self.children = []


class ID3:
    def entropy(self, prob):
        if prob == 1 or prob == 0: 
            return 0
        return - prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)
    

    def classify(self, x):
        if x > 0.5:
            return self.marks[0]
        return self.marks[1]
    

    def find_feature(self, x, y):
        features = x.T
        etr_total = self.entropy(len(x[y == self.marks[0]]) / len(x))
        i_gains = [0] * self.n_features

        for i in range(self.n_features):
            values = list(set(features[i]))
            for j in range(len(values)):
                arr_x = x[features[i] == values[j]]
                arr_y = y[features[i] == values[j]]
                prob = len(arr_x) / len(x)
                mark_prob = len(arr_x[arr_y == self.marks[0]]) / len(arr_x)
                etr = self.entropy(mark_prob)
                i_gains[i] += prob * etr
            i_gains[i] = etr_total - i_gains[i]
        return np.argmax(i_gains)
    

    def branch(self, node: Node, x, y, height):
        prob = len(x[y == self.marks[0]]) / len(x)
        node.mark = self.classify(prob)

        if len(x[y == self.marks[0]]) == len(x):
            return
        if len(x[y == self.marks[0]]) == 0:
            return
        if height > self.max_height:
            return 
        
        index = self.find_feature(x, y)
        node.index = index
        features = list(set(x.T[index]))
        for feature in features:
            n = Node()
            n.feature = feature
            self.branch(n, x[x.T[index] == feature], y[x.T[index] == feature], height + 1)
            node.children.append(n)
    

    def fit(self, x, y):
        self.tree = Node()
        self.x = x
        self.y = y

        self.marks = list(set(y))

        self.size = x.shape[0]
        self.n_features = x.shape[1]

        self.max_height = 3
    
    
    def train(self, max_height = 3):
        self.max_height = max_height
        self.tree = Node()
        self.branch(self.tree, self.x, self.y, 0)


    def predict(self, x):
        node = self.tree
        while len(node.children) != 0:
            for e in node.children:
                if x[node.index] == e.feature:
                    node = e
                    break
        return node.mark 

    
def tree_runner(node: Node):
    if node == None:
        return
    print("NODE")
    print(f" mark: {node.mark}")
    print(f" feature: {node.feature}")
    for n in node.children:
        tree_runner(n)