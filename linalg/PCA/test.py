import numpy as np
from matplotlib import pyplot as plt
from model import PCA 

cov = np.array([
     [5, 0],
     [0, 1]
])

N = 30

deg = 0 / 180 * np.pi

rot = np.array([
    [np.cos(deg),  -np.sin(deg)],
    [np.sin(deg), np.cos(deg)]
])

data = np.array(list(zip(np.random.normal(0, 1, N), np.random.normal(0, 1, N))))

data = data.dot(cov).dot(rot)


pca = PCA()
pca.fit(data)

new_x = pca.eval()
new_x = new_x[:,0]
new_data = np.array(list(zip(new_x, [0 for _ in range(N)])))
plt.axis("equal")
plt.plot(data.T[0], data.T[1], "o")
plt.plot(new_data.T[0], new_data.T[1], 'o')
plt.show()

