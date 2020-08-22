import numpy as np
a = np.array([[1, 2], [3, 4]])
b = a.view()
c = a
d = a.copy()  # a new copy of a , pure copy
print(a, b, c, sep='\n')
a[0][0] = 100
print("stage2", a, b, c,d, sep='\n')
b.shape = (4, -1)
print("stage3", a, b, c,d, sep='\n')
c.shape = (1, 4)
print("stage4", a, b, c,d, sep='\n')
