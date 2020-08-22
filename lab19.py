import matplotlib.pyplot as plt
import numpy as np

w = 3.0
B = [-8, 0, 8]
L = ['b=-8', 'b=0', 'b=8']
x = np.arange(-10, 10, 0.1)
for b, l in zip(B, L):
    f = 1 / (1 + np.exp(-(w * x + b)))
    plt.plot(x, f, label=l)
plt.legend(loc=2)
plt.show()
