import numpy as np
from matplotlib import pyplot as plt
import random

class K_means:
    def __init__(self, data):
        self.data = data
        self.n = 3

    def init_groups(self):
        self.groups = [[] for _ in range(self.n)]
        self.centeroids = [None for _ in range(self.n)]
        index = 0
        for element in self.data:
            self.groups[index].append(element)
            index += 1
            if index >= self.n:
                index = 0 

    def get_centeroids(self):
        for index, group in enumerate(self.groups):
            self.centeroids[index] = 0
            for i in range(len(group)):
                self.centeroids[index] += group[i]
            self.centeroids[index] /= len(group)

    def get_groups(self):
        self.groups = [[] for _ in range(self.n)]
        self.distances = [None for _ in range(self.n)]
        index = 0
        for element in self.data:
            for i in range(self.n):
                self.distances[i] = self.get_distance(element, self.centeroids[i])
            index = np.argmin(self.distances)
            self.groups[index].append(element)

    def get_distance(self, p1, p2):
        return np.sqrt(np.square(np.abs(p1 - p2)).sum())
    
    def evaluate(self, n, iters):
        self.n = n
        self.init_groups()
        for _ in range(iters):
            self.get_centeroids()
            self.get_groups()
        return self.groups

X = []
X.extend(np.array([[float(random.uniform(0, 7)), random.uniform(1, 8)] for _ in range(20) ]))
X.extend(np.array([[float(random.uniform(8, 12)), random.uniform(5, 8)] for _ in range(20) ]))
X.extend(np.array([[float(random.uniform(0, 5)), random.uniform(5, 8)] for _ in range(20) ]))
print(X)

alg = K_means(X)
for i in range(20):
    groups = alg.evaluate(3, i)
    plt.plot(np.array(groups[0])[:,0], np.array(groups[0])[:,1], 'o', color="red")
    plt.plot(np.array(groups[1])[:,0], np.array(groups[1])[:,1], 'o', color="blue")
    plt.plot(np.array(groups[2])[:,0], np.array(groups[2])[:,1], 'o', color="green")
    plt.savefig(f"{i}.png")