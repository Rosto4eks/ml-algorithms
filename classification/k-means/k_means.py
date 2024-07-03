import numpy as np
from matplotlib import pyplot as plt

class K_means:
    def __init__(self, data):
        self.data = np.array(data)
        self.n = 3
        self.shape = self.data.shape[1]

    def init_groups(self):
        self.groups = [[] for _ in range(self.n)]
        self.centeroids = [[0 for _ in range(self.shape)] for _ in range(self.n)]
        index = 0
        for element in self.data:
            self.groups[index].append(element)
            index += 1
            if index >= self.n:
                index = 0 
        print(self.groups[0])

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