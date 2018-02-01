import numpy as np

# a = np.ones((2,3,4,4))
# m = np.arange(16).reshape((4,4))
# # x = np.rot90(m,2, (2,3))
# print(m)
# print(m[1:-1,1:-1])
model_list = (lambda olst, glst: [ model_list(item, glst) if type(item) is list else glst.pop(0) for item in olst])
A=[1,[2,3],[4,[5],6],7]
B=[2,3,4,5,6,7,8]
print(model_list(A,B))