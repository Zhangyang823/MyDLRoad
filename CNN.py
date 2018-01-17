import numpy as np
import sys

def  Conv2( X, k ):
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
    ret = np.empty((ret_row, ret_col))
    for i in range( ret_row ):
        for j in range( ret_col ):
            ret[i,j] = np.sum(X[i:i+k_row,j:j+k_col] * k )
    return ret

def rot180( X ):
    x_row, x_col = X.shape
    ret_row, ret_col = x_row, x_col
    ret = X.copy()
    for i in range( ret_row ):
        for j in range( int(ret_col / 2) ):
            temp = ret[i,j]
            ret[i,j] = ret[i,ret_col-j - 1]
            ret[i,ret_col-j - 1] = temp
    for j in range( ret_col ):
        for i in range( int(ret_row / 2) ):     
            temp = ret[i,j]
            ret[i,j] = ret[ret_row - i - 1,j]
            ret[ret_row - i - 1,j] = temp
    return ret



X = np.array([[1,2,3,4],
            [5,6,7,8],
            [3,1,6,4],
            [4,2,3,4]])
k = np.array([[1,2],[1,2]])

print( Conv2(X,k))
print( rot180(X))
