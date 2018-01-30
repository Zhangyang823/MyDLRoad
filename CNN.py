import numpy as np
import time
import sys
import random
# from tt import TwoLayerNet

def conv2(X, k):
    # as a demo code, here we ignore the shape check
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
    ret = np.empty((ret_row, ret_col))
    for y in range(ret_row):
        for x in range(ret_col):
            sub = X[y: y + k_row, x: x + k_col]
            ret[y, x] = np.sum(sub * k)
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
    ret[size:cur_r + size, size:cur_w + size] = in_data
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
        self.gradient_w = np.zeros_like(self.w, dtype=np.float64)
        for b_id in range(in_batch):
            for i in range(in_channel):
                for o in range(out_channel):
                    self.gradient_w[i, o] += conv2(self.bottom_val[b_id, i], residual[b_id, o])
        # self.gradient_w /= self.batch_size
        # self.gradient_w /= in_batch
        # gradient_x
        gradient_x = np.zeros_like(self.bottom_val, dtype=np.float64)
        for b_id in range(in_batch):
            for i in range(in_channel):
                for o in range(out_channel):
                    gradient_x[b_id, i] += conv2(padding(residual[b_id, o], kernel_size - 1), rot180(self.w[i, o]))
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
    def __init__(self, in_num, out_num, lr=0.01, momentum=0.9, std = 1e-4, reg = 0.75):
        self._in_num = in_num
        self._out_num = out_num
        self.w =std * np.random.randn(in_num, out_num)
        self.b =std * np.zeros(out_num)
        self.lr = lr
        self.momentum = momentum
        self.prev_grad_w = np.zeros_like(self.w)
        self.prev_grad_b = np.zeros_like(self.b)
        self.reg = reg

    # def _sigmoid(self, in_data):
    #     return 1 / (1 + np.exp(-in_data))
    def forward(self, in_data):
        self.bottom_val = in_data
        self.top_val = in_data.dot(self.w) + self.b
        return self.top_val

    def backward(self, loss):
        # in_batch = loss.shape[0]
        #
        # # residual_z = loss * self.topVal * (1 - self.topVal)
        # grad_w = np.zeros_like(self.w, dtype=np.float64)
        # for b_id in range(in_batch):
        #     grad_w += np.dot(loss[b_id].reshape(grad_w.shape[0], 1),
        #                      self.bottom_val[b_id].reshape(1, grad_w.shape[1])).reshape(grad_w.shape)
        # grad_b = np.sum(loss)
        # residual_x = np.zeros_like(self.bottom_val, dtype=np.float64)
        # for b_id in range(in_batch):
        #     # aaa = np.dot(self.w, loss[b_id].T)
        #     residual_x[b_id] = np.dot(self.w.T, loss[b_id])
        # # residual_x = np.dot(self.w, loss)
        # self.prev_grad_w = self.prev_grad_w * self.momentum - grad_w
        # self.prev_grad_b = self.prev_grad_b * self.momentum - grad_b
        # # self.w -= self.lr * self.prev_grad_w
        # # self.b -= self.lr * self.prev_grad_b
        # self.w -= self.lr * grad_w
        # self.b -= self.lr * grad_b

        residual_x = loss.dot(self.w.T)
        self.w -= self.lr *  (self.bottom_val.T.dot(loss) + self.prev_grad_w * self.reg)
        self.b -= self.lr * (np.sum(loss, axis=0))
        self.prev_grad_w = self.w
        self.prev_grad_b = self.b
        return residual_x


class ReLULayer:
    def __init__(self, name='ReLU'):
        pass

    def forward(self, in_data):

        in_data[in_data<0] = 0
        self.top_val = in_data
        return in_data

    def backward(self, residual):
        return (self.top_val > 0) * residual


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
                        offset_r, offset_c = np.where(
                            self.flag[b_id, c, oy * k: oy * k + height, ox * k: ox * k + width] == 1)
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
        shift_scores = in_data - np.max(in_data, axis=1).reshape(-1, 1)
        self.top_val = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        return self.top_val

    def backward(self, residual):
        N = residual.shape[0]
        dscores = self.top_val.copy()
        dscores[range(N), list(residual)] -= 1
        dscores /= N
        # for i in range(N):
        #     self.top_val[i, residual[i]] -= 1
        # self.top_val /= N
        return dscores


class Net:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def train(self, trainData, trainLabel, validData, validLabel, batch_size, iteration):
        train_num = trainData.shape[0]
        for iter in range(iteration):
            print(str(time.clock()) + '  iter=' + str(iter))
            for batch_iter in range(0, train_num, batch_size):
                if batch_iter + batch_size < train_num:
                    loss = self.train_inner(trainData[batch_iter: batch_iter + batch_size],
                                     trainLabel[batch_iter: batch_iter + batch_size])
                else:
                    loss = self.train_inner(trainData[batch_iter: train_num],
                                     trainLabel[batch_iter: train_num])
                print(str(batch_iter) + '/' + str(train_num) + '   loss : ' + str(loss))
            print(str(time.clock()) + "  eval=" + str(self.eval(trainData, trainLabel)))
            print(str(time.clock()) + "  eval=" + str(self.eval(validData, validLabel)))

    def train_inner(self, data, label):
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        N = out_data.shape[0]
        loss = -np.sum(np.log(out_data[range(N), list(label)]))
        loss /= N
        residual_in = label
        for i in range(0, lay_num):
            residual_out = self.layers[lay_num - i - 1].backward(residual_in)
            residual_in = residual_out
        return loss

    def eval(self, data, label):
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        out_idx = np.argmax(in_data, axis=1)
        # label_idx = np.argmax(label, axis=1)
        label_idx = label
        return np.sum(out_idx == label_idx) / float(out_idx.shape[0])


import struct
from array import array


def loadImageSet(filename):
    print("load image set", filename)
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    print("head,", head)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, -1])
    print("load imgs finished")
    return imgs


def loadLabelSet(filename):
    print("load label set", filename)
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    print("head,", head)
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    # ret = np.zeros((imgNum, 10), dtype=np.float64)
    # for i in range(imgNum):
    #     ret[i][labels[i]] = 1.0
    print('load label finished')
    ret = labels
    return ret


# train_feature = loadImageSet("data//MNIST_data//train-images.idx3-ubyte")
# train_feature[train_feature > 0] = 1
# train_label = loadLabelSet("data//MNIST_data//train-labels.idx1-ubyte")
# valid_feature = loadImageSet("data//MNIST_data//t10k-images.idx3-ubyte")
# valid_feature[valid_feature > 0] = 1
# valid_label = loadLabelSet("data//MNIST_data//t10k-labels.idx1-ubyte")
# train_feature = loadImageSet("data\\MNIST_data\\train-images.idx3-ubyte")
# train_label = loadLabelSet("data\\MNIST_data\\train-labels.idx1-ubyte")
# valid_feature = loadImageSet("data\\MNIST_data\\t10k-images.idx3-ubyte")
# valid_label = loadLabelSet("data\\MNIST_data\\t10k-labels.idx1-ubyte")

net = Net()
net.addLayer(ConvLayer(3, 8, 4, 1, 0.9))
net.addLayer(ReLULayer())
net.addLayer(MaxPoolingLayer(2))

net.addLayer(ConvLayer(8, 5, 5, 1, 0.9))
net.addLayer(ReLULayer())
net.addLayer(MaxPoolingLayer(3))

net.addLayer(FlattenLayer())
net.addLayer(FCLayer(4 * 4 *5, 50, 1, 0.9))
# net.addLayer(FCLayer(300, 100, 0.1, 0.9))
net.addLayer(ReLULayer())
net.addLayer(FCLayer(50, 10, 1, 0.9))
net.addLayer(SoftmaxLayer())
# print('net build ok')
from data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    # cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    cifar10_dir = 'D:DML\github\MyDLRoad\cs231n\datasets\cifar-10-batches-py'

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    # X_train = X_train.reshape(num_training, -1)
    X_train = np.transpose(X_train,[0,3,1,2])
    X_val = np.transpose(X_val, [0,3,1,2])
    # X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
N = 100
net.train(X_train[0:N], y_train[0:N], X_val[0:N], y_val[0:N],10,10)
# net.train(train_feature, train_label, valid_feature[0:100], valid_label[0:100], 10 ,10)