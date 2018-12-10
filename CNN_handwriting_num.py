'''
simple CNN model by tensorflow(need the file "MNIST_data_set" in local path)
net model:
    [all active function = Relu , output function = softmax]
    [cross entropy loss: -y_predict * log(y_train)]
    28x28(origin image) - 5x5size/1channel/32amount/1stride/Same padding(convolution filter) - 2x2size/2stride(max_pool) ->
    14x14size/32channel(image 1) - 5x5/32channel/64amount/1stride/same padding(convolution filter) - 2x2size/2stride(max_pool) ->
    7x7size/64channel(image 2) - fully connected(layer1:7x7x64,layer2:1024) - drop_out - soft_max - arg_max

tips:
    batch_size prefer 8X,benefit to GPU calculation
    std_dev will lead to nan weight(>0.1),will reduce proportion of bad weight initial(can't fit)
    batch size will strongly affect accuracy(need more try to find this para)[now = 50]
    some time got run bug,try to restart it

update record:
    with GPU(GTX1060),test data accuracy:0.991,time used:58s
    add show model for visualization of every layer output. 2018/12/10
'''

import time
import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# hyper paras and config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '-1':only CPU, '0':only GPU
save_path = './save_data/CNN_handwriting_num/' + 'train_paras.ckpt'  # save train paras data
is_train = True  # True:train ,False:predict
echo = 10
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


def show_layer_image(input_image):
    # for show convolution layer
    # need input_image.shape = (size,size,num)
    num = input_image.shape[2]
    resize_size = 100  # size of every small image
    b = []
    for vertical in range(int(np.ceil(num/9.))):
        a = []
        if np.floor(num/9.) <= vertical:
            iter_n = num % 9
            for i in range(iter_n):
                # use Interpolation method to resize image
                image_new = cv2.resize(input_image[:, :, i + vertical * 9], (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
                a.append(image_new)
            for i in range(9 - iter_n):
                a.append(np.zeros([resize_size,resize_size]))
            b.append(np.hstack(a))
        else:
            iter_n = 9
            for i in range(iter_n):
                image_new = cv2.resize(input_image[:, :, i + vertical * 9], (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
                a.append(image_new)
            b.append(np.hstack(a))

    return np.vstack(b)



# cnn model
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

# output layer and soft_max
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

# predict
predict = tf.argmax(y_conv, axis=1)

# calc iter
train_images_num = mnist.train.images.shape[0]
iter_num = np.int32(train_images_num / batch_size * echo)

# fit
if is_train == True:
    for i in range(iter_num):

        batch = mnist.train.next_batch(batch_size)  # batch = 8 X ,can improve GPU calculation

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1], keep_proportion: 1.0})
            print("iter: %d, training accuracy %f" % (i, train_accuracy))

        sess.run(train_model,feed_dict={x: batch[0], y_: batch[1], keep_proportion: 0.5})

    saver.save(sess,save_path)
    print("train time used:", time.time() - time_start)
    print("test accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_proportion: 1.0}))
else:
    saver.restore(sess,save_path)
    print("predict time used:",time.time() - time_start)

# funny input
funny_input = plt.imread('./test_data_set/7.png')[:,:,1]
funny_input_ = funny_input.reshape([1,784])

# # predict
# print("test accuracy %f" % sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_proportion:1.0}))  # predict test,not dropout
# predict_list = sess.run(predict,feed_dict={x:funny_input_,keep_proportion:1.0})
# print("predict:",predict_list)
# plt.matshow(funny_input.reshape(28,28))
# plt.title("predict:"+str(predict_list))
# plt.show()

# show model
'show origin image'
cv2.imshow("origin image",cv2.resize(funny_input,(200,200)))

'show layer of convolution 1'
conv_layer_1 = h_conv1[0]  # h_conv1.shape=[-1,x,x,1]
# conv_layer_1_output = sess.run(conv_layer_1,feed_dict={x:mnist.test.images})  # show test image
conv_layer_1_output = sess.run(conv_layer_1,feed_dict={x:funny_input_})  # show funny_input_
cv2.imshow("layer of convolution 1",show_layer_image(conv_layer_1_output))

'show layer of convolution 2'
conv_layer_1 = h_conv2[0]
conv_layer_1_output = sess.run(conv_layer_1,feed_dict={x:funny_input_})
cv2.imshow("layer of convolution 2",show_layer_image(conv_layer_1_output))

cv2.waitKey(0)
