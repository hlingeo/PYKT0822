import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.1)
f = 1 / (1 + np.exp(-x))
plt.xlabel(x)
plt.ylabel('g(z)')
plt.plot(x, f)
plt.grid()
plt.show()
