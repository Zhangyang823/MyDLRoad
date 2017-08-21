#============================================================================
#图像分类实验
#数据采用CIFAR-10
#============================================================================
import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

#============================================================================
#下载数据
#============================================================================
cifar10.maybe_download_and_extract()

#============================================================================
#初始化weight函数，但是为了防止因为特征过多而引起的过拟合，给每个weight加一个L2的loss。
#一般来说L1正则会造成稀疏的特征，大部分无用特征会被置为0。L2特征会让权重不过大，是权重比较平均。
#函数中用w1来控制loss的大小
#============================================================================
def variable_with_weight_loss( shape, stddev, w1 ):
    var = tf.Variable( tf.truncated_normal( shape, stddev = stddev ) )
    if w1 is not None:
        weight_loss = tf.multiply( tf.nn.l2_loss( var ), w1, name = 'weight_loss' )
        tf.add_to_collection( 'losses', weight_loss )
    return var

#============================================================================
#data
#为了训练增加数据，cifar10_input.distorted_inputs函数会增加包括以下操作
#1、tf.random_crop:随机剪切一块大小 24 × 24 大小的图片
#2、tf.image.random_flip_left_right：随机水平翻转
#3、tf.image.random_brightness：随机设置亮度
#4、tf.image.random_contrast：随机设置对比度
#为了加速处理，distorted_inputs使用了16个独立线程来加速任务
#
#测试数据不需要太多，只使用了tf.random_crop
#============================================================================
images_train, labels_train = cifar10_input.distorted_inputs( data_dir = data_dir, batch_size = batch_size )
image_test, labels_test = cifar10_input.inputs( eval_data = True, data_dir = data_dir, batch_size = batch_size )

#============================================================================
#placeholder
#============================================================================
image_holder = tf.placeholder( tf.float32, [batch_size, 24, 24, 3 ] )
label_holder = tf.placeholder( tf.int32, [batch_size] )

#============================================================================
#first layer
#一个卷积、一个relu激活、一个max_pool池化、一个lrn
#lrn模仿了生物系统的“侧抑制”机制，对局部神经元的活动创造竞争环境，使得其中响应值更大，并抑制
#反馈较小的神经元lrn能增强模型的泛华能力。lrn对ReLU这种没有上界的激活函数比较有用，但是对与
#Sigmoid这种有固定边界并且能抑制过大值的激活函数作用不大。
#============================================================================
weight1 = variable_with_weight_loss( shape = [5, 5, 3, 64 ], stddev = 5e-2, w1 = 0.0 )
kernel1 = tf.nn.conv2d( image_holder, weight1, [ 1, 1, 1, 1 ], padding = 'SAME' )
bias1 = tf.Variable( tf.constant( 0.0, shape = [64] ) )
conv1 = tf.nn.relu( tf.nn.bias_add( kernel1, bias1 ) )
pool1 = tf.nn.max_pool( conv1, ksize = [ 1, 3, 3, 1 ], strides = [ 1, 2, 2, 1 ], padding = 'SAME' )
norm1 = tf.nn.lrn( pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75 )

#============================================================================
#sencond layer
#一个卷积、一个relu、一个lrn、一个max_pool。
#============================================================================
weight2 =variable_with_weight_loss( shape = [ 5, 5, 64, 64 ], stddev = 5e-2, w1 = 0.0 )
kernel2 = tf.nn.conv2d( norm1, weight2, [ 1, 1,  1, 1 ], padding = 'SAME' )
bias2 = tf.Variable( tf.constant( 0.1, shape = [64] ) )
conv2 = tf.nn.relu( tf.nn.bias_add( kernel2, bias2 ) )
norm2 = tf.nn.lrn( conv2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75 )
pool2 = tf.nn.max_pool( norm2, ksize = [ 1, 3, 3, 1 ], strides = [ 1, 2, 2, 1 ], 
                       padding = 'SAME' )

#============================================================================
#full connection 1
#============================================================================
reshape = tf.reshape( pool2, [ batch_size, -1 ] )
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss( shape = [ dim, 384 ], stddev = 0.04, w1 = 0.004 )
bias3 = tf.Variable( tf.constant( 0.1, shape = [ 384 ] ) )
local3 = tf.nn.relu( tf.matmul( reshape, weight3 ) + bias3 )

#============================================================================
#full connection 2
#============================================================================
weight4 = variable_with_weight_loss( shape = [ 384, 192 ], stddev = 0.04, w1 = 0.004 )
bias4 = tf.Variable( tf.constant( 0.1, shape = [ 192 ] ) )
local4 = tf.nn.relu( tf.matmul( local3, weight4 ) + bias4 )

#============================================================================
#full connection 3，未使用softmax，在后面的loss计算才使用。
#不使用softmax也能得出结果，只需要比较几个数的大小选最大的即可
#============================================================================
weight5 = variable_with_weight_loss( shape = [ 192, 10 ], stddev = 1 / 192.0, w1 = 0.0 )
bias5 = tf.Variable( tf.constant( 0.0, shape = [ 10 ] ) )
logits = tf.add( tf.matmul( local4, weight5 ), bias5 )
#logits = tf.matmul( local4, weight5 ) + bias5

#============================================================================
#loss function
#============================================================================
def loss( logits, labels ):
    labels = tf.cast( labels, tf.int64 )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( 
            logits = logits, labels = labels, name = 'cross_entropy_per_example' )
    cross_entropy_mean = tf.reduce_mean( cross_entropy, name = 'cross_entropy' )
    tf.add_to_collection( 'losses', cross_entropy_mean )

    return tf.add_n( tf.get_collection( 'losses'), name = 'total_loss' )

#============================================================================
#优化器选择Adam Optimizer 速率为1e-3
#============================================================================
loss = loss( logits, label_holder )    
train_op = tf.train.AdamOptimizer( 1e-3 ).minimize( loss )
top_k_op = tf.nn.in_top_k( logits, label_holder, 1 )

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#============================================================================
#启动线程，因为之前的获取数据需要16个线程，不启动后续的训练无法开始
#============================================================================
tf.train.start_queue_runners()

#============================================================================
#训练
#============================================================================
for step in range( max_steps ):
    start_time = time.time()
    image_batch, laber_batch = sess.run( [ images_train, labels_train ] )
    _, loss_value = sess.run( [ train_op, loss ], 
                             feed_dict = { image_holder : image_batch, 
                                          label_holder : laber_batch } )
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float( duration )

        format_str = ( 'step %d, loss = %.2f ( %.1f examples / sec; %.3f sec / batch ) ' )
        print( format_str% ( step, loss_value, examples_per_sec, sec_per_batch ) )

#============================================================================
#计算准确率
#============================================================================        
num_examples = 10000
import math
num_iter = int( math.ceil( num_examples / batch_size ) )
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, laber_batch = sess.run( [ image_test, labels_test ] )
    predictions = sess.run( [ top_k_op ], feed_dict = { image_holder : image_batch,
                           label_holder : laber_batch } )
    true_count += np.sum( predictions )
    step += 1

precision = true_count / total_sample_count
print( 'precision @ 1 = %.3f' % precision )     