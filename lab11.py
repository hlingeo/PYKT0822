import numpy as np

a = np.zeros((1, 2))
b = np.ones((2, 1))
print(a)
print(b)
#
c = np.zeros((10, 2))
d = c.T
e = d.view()
print(c.shape, d.shape, e.shape)
f = np.reshape(d, (5, 4))
g = np.reshape(d, (20,)) # 1 D
h = np.reshape(d, (20, -1)) # (20,1) 20 row, 1 col 2-dim
i = np.reshape(d, (-1, 20)) # (1,20) 1 row, 20 col
# 2-dim
print(f.shape, g.shape, h.shape, i.shape)