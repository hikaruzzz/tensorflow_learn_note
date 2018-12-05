'''
tips:
    batch_size prefer 8X,benefit to GPU calculation
    std_dev will lead to nan weight(>0.1),will reduce proportion of bad weight initial(can't fit)

record:
    batch size will strongly affect accuracy(need more try to find this para)[now = 50]
'''

import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# hyper paras and config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_path = './save_data/CNN_handwriting_num/' + 'train_paras.ckpt'  # save train paras data
is_train = True  # train or predict
echo = 10000
learn_rate = 0.001
batch_size = 48
std_dev = 0.01  # standard deviation of weight initial
restart_threshold = 0.5
time_start = time.time()
mnist = input_data.read_data_sets('./MNIST_data_set', one_hot=True)


def weight_init(shape):
    #initial = tf.truncated_normal(shape, stddev=std_dev)  # truncated random
    initial = tf.random_normal(shape, mean=0.0, stddev=std_dev)
    return tf.Variable(initial)


def bias_init(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # convolution function
    'The stride of the sliding window for each dimension of `input`'
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# fit model

# placeholder
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])


# first layer
x_image = tf.reshape(x, [-1, 28, 28, 1])  # [batch, in_height, in_width, in_channels]
W_conv1 = weight_init([5, 5, 1, 32])  # filter:[size=5x5,channel=1,filter_amount=32]
b_conv1 = bias_init([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# second layer
W_conv2 = weight_init([5, 5, 32, 64])  # weight_init => Variables, can SGD
b_conv2 = bias_init([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# neural net layer
'y = x * w + b = [-1,7x7x64] * [7x7x64,1024] + [1024]'
W_fc1 = weight_init([7 * 7 * 64, 1024])  # after x2 pool,image size decrease to 7x7,
b_fc1 = bias_init([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# dropout
keep_proportion = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_proportion)


# softmax
W_fc2 = weight_init([1024, 10])
b_fc2 = bias_init([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# loss and optimizer model
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_model = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())  # init variables
saver = tf.train.Saver()

if is_train == True:
    for i in range(echo):

        batch = mnist.train.next_batch(batch_size)  # batch = 8 X ,can improve GPU calculation

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1], keep_proportion: 1.0})
            print("step %d, training accuracy %f" % (i, train_accuracy))

        sess.run(train_model,feed_dict={x: batch[0], y_: batch[1], keep_proportion: 0.5})
    saver.save(sess,save_path)
    print("train time used:", time.time() - time_start)
else:
    saver.restore(sess,save_path)
    print("predict time used:",time.time() - time_start)


print("test accuracy %f" % sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_proportion:1.0}))

# show
print(mnist.test.images.shape)  # (1w,784)
