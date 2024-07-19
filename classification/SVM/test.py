import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from model import LinearSVM


X = np.array(
    [[np.random.normal(1, 0.3), np.random.normal(1, 0.3)] for _ in range(100)] +
    [[np.random.normal(1, 0.3), np.random.normal(3, 0.3)] for _ in range(100)]
)

Y = np.array(
    [1 for _ in range(100)] +
    [-1 for _ in range(100)]
)

data = list(zip(X, Y))
np.random.shuffle(data)

x, y = zip(*data)
x = np.array(x)
y = np.array(y)

model = LinearSVM()
model.fit(x, y)


fig, ax = plt.subplots()
ax.axis("equal")
scatter = ax.scatter(x.T[0], x.T[1], c=y, cmap="bwr")

ax.set_xlim(x[:, 0].min() - 0.1, x[:, 0].max() + 0.1)
ax.set_ylim(x[:, 1].min() - 0.1 , x[:, 1].max() + 0.1)

x1 = -100
x2 = 100

frames = []

def get_y(k):
    return (-model.w[0] * x1 - model.b + k) / model.w[1], (-model.w[0] * x2 - model.b + k) / model.w[1]

for i in range(1000):
    model.train(epoch=1, learning_rate=0.1, batch_size=25)
    line1, = ax.plot([x1, x2], get_y(0), "k-", lw=1)
    line2, = ax.plot([x1, x2], get_y(-1), "k:", lw=1)
    line3, = ax.plot([x1, x2], get_y(1), "k:", lw=1)
    frames.append([line1, line2, line3])

    

anim = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)

plt.show()