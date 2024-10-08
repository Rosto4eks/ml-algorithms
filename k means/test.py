import numpy as np
from matplotlib import pyplot as plt
import random
from model import K_means
import time

colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'purple',
    'orange',
    "darkgreen"
]

X = []
X.extend(np.array([[float(random.uniform(0, 7)), random.uniform(1, 8)] for _ in range(50) ]))
X.extend(np.array([[float(random.uniform(8, 12)), random.uniform(5, 8)] for _ in range(60) ]))
X.extend(np.array([[float(random.uniform(0, 5)), random.uniform(3, 7)] for _ in range(70) ]))

plt.ion()

n_classes = 3
iterations = 10

for i in range(iterations):
    model = K_means(iters=i, max_num=n_classes)
    groups = model.fit(X)
    centroids = model.centeroids

    plt.plot(np.array(centroids)[:, 0], np.array(centroids)[:, 1], 'o', color="black", markersize=12)
    for j in range(n_classes):
        if len(groups[j]) > 0:
            plt.plot(np.array(groups[j])[:,0], np.array(groups[j])[:,1], 'o', color=colors[j])
    plt.draw()
    plt.pause(1)
    plt.clf()