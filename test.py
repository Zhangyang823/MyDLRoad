import tensorflow as tf
import numpy as np 
from tensorflow.models.tutorials.rnn.ptb import reader


def xxxxx(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)#raw data : train_data | vali_data | test data

    data_len = len(raw_data) #how many words in the data_set
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)#batch_len 就是几个word的意思
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    x = data[:, 0*num_steps:(0+1)*num_steps]
    y = data[:, 0*num_steps+1:(0+1)*num_steps+1]
    
    for i in range(epoch_size):
        yield (x, y)
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    

DATA_PATH = "G:\\DML\\github\\tensorflow\\simple-examples\\data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data( DATA_PATH )
#ttt = xxxxx( train_data, 4, 5 )
cnt = 0

for step, ( x, y ) in enumerate(
        xxxxx( train_data, 20, 35 )
    ): 
    cnt += 1
    if cnt % 100 == 0 :
        print( cnt )
print(cnt)
