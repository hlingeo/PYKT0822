import matplotlib.pyplot as plt
import numpy as np

W = [0.5, 1.0, 2.0]
L = ['w=0.5', 'w=1.0', 'w=2.0']
x = np.arange(-10, 10, 0.1)
for w, l in zip(W, L):
    f = 1 / (1 + np.exp(-(w * x + 0)))
    plt.plot(x, f, label=l)
plt.legend(loc=2)
plt.show()
