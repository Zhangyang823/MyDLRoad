import numpy as np

a = np.ones((2,3,4,4))
m = np.arange(16).reshape((4,4))
# x = np.rot90(m,2, (2,3))
print(m)
print(m[1:-1,1:-1])