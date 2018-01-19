import numpy as np
import time
import sys
import random

def conv2(X, k):
    # as a demo code, here we ignore the shape check
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
    ret = np.empty((ret_row, ret_col))
    for y in range(ret_row):
        for x in range(ret_col):
            sub = X[y : y + k_row, x : x + k_col]
            ret[y,x] = np.sum(sub * k)
    return ret

def rot180(in_data):
    ret = in_data.copy()
    yEnd = ret.shape[0] - 1
    xEnd = ret.shape[1] - 1
    for y in range(int(ret.shape[0] / 2)):
        for x in range(ret.shape[1]):
            ret[yEnd - y][x] = ret[y][x]
    for y in range(ret.shape[0]):
        for x in range(int(ret.shape[1] / 2)):
            ret[y][xEnd - x] = ret[y][x]
    return ret

def padding(in_data, size):
    cur_r, cur_w = in_data.shape[0], in_data.shape[1]
    new_r = cur_r + size * 2
    new_w = cur_w + size * 2
    ret = np.zeros((new_r, new_w))
    ret[size:cur_r + size, size:cur_w+size] = in_data
    return ret

def discreterize(in_data, size):
    num = in_data.shape[0]
    ret = np.zeros((num, size))
    for i, idx in enumerate(in_data):
        ret[i, idx] = 1
    return ret

class ConvLayer(object):
    def __init__(self, in_channel, out_channel, kernel_size, lr=0.01, momentum=0.9, name='Conv'):
        self.w = np.random.rand(in_channel, out_channel, kernel_size, kernel_size)
        self.b = np.random.rand(out_channel)
        self.w = self.w / np.sum(self.w)
        self.b = self.b / np.sum(self.b)
        self.layer_name = name
        self.lr = lr
        self.momentum = momentum

        self.prev_gradient_w = np.zeros_like(self.w)
        self.prev_gradient_b = np.zeros_like(self.b)
    # def _relu(self, x):
    #     x[x < 0] = 0
    #     return x
    def forward(self, in_data):
        # assume the first index is channel index
        # print ('conv forward:' + str(in_data.shape))
        in_batch, in_channel, in_row, in_col = in_data.shape
        out_channel, kernel_size = self.w.shape[1], self.w.shape[2]
        self.top_val = np.zeros((in_batch, out_channel, in_row - kernel_size + 1, in_col - kernel_size + 1))
        self.bottom_val = in_data

        for b_id in range(in_batch):
            for o in range(out_channel):
                for i in range(in_channel):
                    self.top_val[b_id, o] += conv2(in_data[b_id, i], self.w[i, o])
                self.top_val[b_id, o] += self.b[o]
        return self.top_val

    def backward(self, residual):
        in_channel, out_channel, kernel_size, _ = self.w.shape
        in_batch = residual.shape[0]
        # gradient_b
        # self.gradient_b = residual.sum(axis=3).sum(axis=2).sum(axis=0) / self.batch_size
        self.gradient_b = residual.sum(axis=3).sum(axis=2).sum(axis=0)
        # gradient_w
        self.gradient_w = np.zeros_like(self.w,dtype=np.float64)
        for b_id in range(in_batch):
            for i in range(in_channel):
                for o in range(out_channel):
                    self.gradient_w[i, o] += conv2(self.bottom_val[b_id,i], residual[b_id,o])
        # self.gradient_w /= self.batch_size
        # self.gradient_w /= in_batch
        # gradient_x
        gradient_x = np.zeros_like(self.bottom_val,dtype=np.float64)
        for b_id in range(in_batch):
            for i in range(in_channel):
                for o in range(out_channel):
                    gradient_x[b_id, i] += conv2(padding(residual[b_id,o], kernel_size - 1), rot180(self.w[i, o]))
        # gradient_x /= self.batch_size
        # gradient_x /= in_batch
        # update
        # self.prev_gradient_w = self.prev_gradient_w * self.momentum - self.gradient_w
        # self.w += self.lr * self.prev_gradient_w
        # self.prev_gradient_b = self.prev_gradient_b * self.momentum - self.gradient_b
        # self.b += self.lr * self.prev_gradient_b
        self.prev_gradient_w = self.prev_gradient_w * self.momentum - self.gradient_w
        self.w += self.lr * self.gradient_w
        self.prev_gradient_b = self.prev_gradient_b * self.momentum - self.gradient_b
        self.b += self.lr * self.gradient_b
        return gradient_x

class FCLayer:
    def __init__(self, in_num, out_num, lr = 0.01, momentum=0.9):
        self._in_num = in_num
        self._out_num = out_num
        self.w = np.random.rand(in_num, out_num)
        # self.w = self.w / np.sum(np.sum(self.w))
        self.b = np.random.rand(out_num, 1)
        # self.b = self.b / np.sum(np.sum(self.b))
        self.w = self.w / np.sum(self.w)
        self.b = self.b / np.sum(self.b)
        self.lr = lr
        self.momentum = momentum
        self.prev_grad_w = np.zeros_like(self.w)
        self.prev_grad_b = np.zeros_like(self.b)
    # def _sigmoid(self, in_data):
    #     return 1 / (1 + np.exp(-in_data))
    def forward(self, in_data):
        # print( 'fc forward=' + str(in_data.shape) )
        in_batch = in_data.shape[0]
        self.top_val = np.zeros((in_batch, self._out_num))
        self.bottom_val = in_data
        for b_id in range(in_batch):
            self.top_val[b_id] = (np.dot(in_data[b_id],self.w ) + self.b.T)
        return self.top_val
    def backward(self, loss):
        in_batch = loss.shape[0]

        # residual_z = loss * self.topVal * (1 - self.topVal)
        grad_w = np.zeros_like(self.w,dtype=np.float64)
        for b_id in range(in_batch):
            grad_w += np.dot(self.bottom_val[b_id].reshape(grad_w.shape[0],1), loss[b_id].reshape(1,grad_w.shape[1])).reshape(grad_w.shape)
        grad_b = np.sum(loss)
        residual_x = np.zeros_like(self.bottom_val,dtype=np.float64)
        for b_id in range(in_batch):
            # aaa = np.dot(self.w, loss[b_id].T)
            residual_x[b_id] = np.dot(self.w, loss[b_id].T).T
        # residual_x = np.dot(self.w, loss)
        self.prev_grad_w = self.prev_grad_w * self.momentum - grad_w
        self.prev_grad_b = self.prev_grad_b * self.momentum - grad_b
        # self.w -= self.lr * self.prev_grad_w
        # self.b -= self.lr * self.prev_grad_b
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
        return residual_x

class ReLULayer:
    def __init__(self, name='ReLU'):
        pass

    def forward(self, in_data):
        self.top_val = in_data
        ret = in_data.copy()
        ret[ret < 0] = 0
        # ret[ret > 0] = 1
        return ret
    def backward(self, residual):
        gradient_x = residual.copy()
        gradient_x[self.top_val < 0] = 0
        gradient_x[self.top_val > 0] = 1
        return gradient_x

class MaxPoolingLayer:
    def __init__(self, kernel_size, name='MaxPool'):
        self.kernel_size = kernel_size

    def forward(self, in_data):
        in_batch, in_channel, in_row, in_col = in_data.shape
        k = self.kernel_size
        out_row = int(in_row / k + (1 if in_row % k != 0 else 0))
        out_col = int(in_col / k + (1 if in_col % k != 0 else 0))

        self.flag = np.zeros_like(in_data)
        ret = np.empty((in_batch, in_channel, out_row, out_col))
        for b_id in range(in_batch):
            for c in range(in_channel):
                for oy in range(out_row):
                    for ox in range(out_col):
                        height = k if (oy + 1) * k <= in_row else in_row - oy * k
                        width = k if (ox + 1) * k <= in_col else in_col - ox * k
                        idx = np.argmax(in_data[b_id, c, oy * k: oy * k + height, ox * k: ox * k + width])
                        offset_r = int(idx / width)
                        offset_c = int(idx % width)
                        self.flag[b_id, c, oy * k + offset_r, ox * k + offset_c] = 1
                        ret[b_id, c, oy, ox] = in_data[b_id, c, oy * k + offset_r, ox * k + offset_c]
        return ret
    def backward(self, residual):
        in_batch, in_channel, in_row, in_col = self.flag.shape
        k = self.kernel_size
        out_row, out_col = residual.shape[2], residual.shape[3]

        gradient_x = np.zeros_like(self.flag)
        for b_id in range(in_batch):
            for c in range(in_channel):
                for oy in range(out_row):
                    for ox in range(out_col):
                        height = k if (oy + 1) * k <= in_row else in_row - oy * k
                        width = k if (ox + 1) * k <= in_col else in_col - ox * k
                        offset_r, offset_c = np.where(self.flag[b_id, c, oy * k : oy * k + height, ox * k : ox * k + width]==1)
                        gradient_x[b_id, c, oy * k + offset_r, ox * k + offset_c] = residual[b_id, c, oy, ox]
        gradient_x[self.flag == 0] = 0
        return gradient_x

class FlattenLayer:
    def __init__(self, name='Flatten'):
        pass
    def forward(self, in_data):
        self.in_batch, self.in_channel, self.r, self.c = in_data.shape
        return in_data.reshape(self.in_batch, self.in_channel * self.r * self.c)
    def backward(self, residual):
        return residual.reshape(self.in_batch, self.in_channel, self.r, self.c)

class SoftmaxLayer:
    def __init__(self, name='Softmax'):
        pass
    def forward(self, in_data):
        self.mm = np.max(in_data, axis=1).reshape((in_data.shape[0],1)).repeat(in_data.shape[1], axis=1)
        self.mm[self.mm<=0] = 1
        in_data = in_data / self.mm
        exp_out = np.exp(in_data)
        self.top_val = exp_out / np.sum(exp_out, axis=1).reshape((exp_out.shape[0],1)).repeat(in_data.shape[1], axis=1)
        return self.top_val
    def backward(self, residual):
        return (self.top_val - residual) * (self.top_val - residual)

class Net:
    def __init__(self):
        self.layers = []
    def addLayer(self, layer):
        self.layers.append(layer)
    def train(self, trainData, trainLabel, validData, validLabel, batch_size, iteration):
        tempTrainData = trainData
        temptrainLabel = trainLabel
        index = random.sample([i for i in range(tempTrainData.shape[0])], 100)
        trainData = tempTrainData[index]
        trainLabel = temptrainLabel[index]
        train_num = trainData.shape[0]
        for iter in range(iteration):

            print (str(time.clock()) + '  iter=' + str(iter) )
            for batch_iter in range(0, train_num, batch_size):
                print(str(batch_iter) + '/' + str(train_num))
                if batch_iter + batch_size < train_num:
                    self.train_inner(trainData[batch_iter: batch_iter + batch_size],
                        trainLabel[batch_iter: batch_iter + batch_size])
                else:
                    self.train_inner(trainData[batch_iter: train_num],
                        trainLabel[batch_iter: train_num])
            print (str(time.clock()) + "  eval=" + str(self.eval(trainData, trainLabel)))
            print(str(time.clock()) + "  eval=" + str(self.eval(validData, validLabel)))
    def train_inner(self, data, label):
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        residual_in = label
        for i in range(0, lay_num):
            residual_out = self.layers[lay_num-i-1].backward(residual_in)
            residual_in = residual_out
    def eval(self, data, label):
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        out_idx = np.argmax(in_data, axis=1)
        label_idx = np.argmax(label, axis=1)
        return np.sum(out_idx == label_idx) / float(out_idx.shape[0])


import struct
from array import array

def loadImageSet(filename):  
    print ("load image set",filename)  
    binfile= open(filename, 'rb')  
    buffers = binfile.read()  
   
    head = struct.unpack_from('>IIII' , buffers ,0)  
    print ("head,",head)  
   
    offset = struct.calcsize('>IIII')  
    imgNum = head[1]  
    width = head[2]  
    height = head[3]  
    #[60000]*28*28  
    bits = imgNum * width * height  
    bitsString = '>' + str(bits) + 'B' #like '>47040000B'  
   
    imgs = struct.unpack_from(bitsString,buffers,offset)  
   
    binfile.close()  
    imgs = np.reshape(imgs,[imgNum,1,width,height])
    print ("load imgs finished")  
    return imgs  
   
def loadLabelSet(filename):  
   
    print ("load label set",filename  )
    binfile = open(filename, 'rb')  
    buffers = binfile.read()  
   
    head = struct.unpack_from('>II' , buffers ,0)  
    print ("head,",head)  
    imgNum=head[1]  
   
    offset = struct.calcsize('>II')  
    numString = '>'+str(imgNum)+"B"  
    labels = struct.unpack_from(numString , buffers , offset)  
    binfile.close()  
    labels = np.reshape(labels,[imgNum,1])

    ret = np.zeros((imgNum,10),dtype=np.float64)
    for i in range(imgNum):
        ret[i][labels[i]] = 1.0
    print ('load label finished'  )
    return ret

train_feature = loadImageSet("data//MNIST_data//train-images.idx3-ubyte")
train_feature[train_feature>0] = 1
train_label = loadLabelSet("data//MNIST_data//train-labels.idx1-ubyte")
valid_feature = loadImageSet("data//MNIST_data//t10k-images.idx3-ubyte")
valid_feature[valid_feature>0] = 1
valid_label = loadLabelSet("data//MNIST_data//t10k-labels.idx1-ubyte")
# train_feature = loadImageSet("data\\MNIST_data\\train-images.idx3-ubyte")
# train_label = loadLabelSet("data\\MNIST_data\\train-labels.idx1-ubyte")
# valid_feature = loadImageSet("data\\MNIST_data\\t10k-images.idx3-ubyte")
# valid_label = loadLabelSet("data\\MNIST_data\\t10k-labels.idx1-ubyte")

net = Net()
net.addLayer(ConvLayer(1, 6, 4, 0.1, 0.9))
net.addLayer(ReLULayer())
net.addLayer(MaxPoolingLayer(2))

net.addLayer(ConvLayer(6, 16, 5, 0.1, 0.9))
net.addLayer(ReLULayer())
net.addLayer(MaxPoolingLayer(3))

net.addLayer(FlattenLayer())
net.addLayer(FCLayer(16 * 3 * 3, 100, 0.1, 0.9))
net.addLayer(ReLULayer())
net.addLayer(FCLayer(100, 10, 0.1, 0.9))
net.addLayer(SoftmaxLayer())
print( 'net build ok')
net.train(train_feature, train_label, valid_feature[0:100], valid_label[0:100], 10 ,10)