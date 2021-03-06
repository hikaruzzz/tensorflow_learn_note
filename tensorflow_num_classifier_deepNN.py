'''
hand writing num classifier
introduce:
     variable easily layer neural net,using SGDoptimizer,datasets from sklearn(1300+simple)
update:
    train.saver and train.restore to save variables
    change model for multiply layer neural net
    activation function: nn.relu , sigmoid ,
tips:
    care learn rate, 0.01 -> 0.0001 more try
    weight[n].shape = (num_layer[n], num_layer[n-1])
    dont't reshape data or argmax() between graph calculation(prefer preprocess all data before train model)
    argmax(...,axis = 0 or 1)   0 is vertical calc, 1 is horizon calc
    nan data: use log_softmax() or reduce learn_rate,like:0.001
record:
    2018.12.5:use gpu but accuracy reduce to 0.6 ,cpu is 0.92,[solved:learn_rate reduce to 0.0001]
    2018.12.4:multiply hide layer,hide layer x2,neural num = 9 and 9,accuracy:0.91778,time used:13.534s(only CPU:E3 1230v2),simple data:1300+,learn_rate=0.001,iter=10000,
    2018.11.25:one hide layer,4 neural，accuracy:0.869,time used:13.2s(only CPU:E3 1230v2),simple data:1300+，learn_rate=0.001,iter=10000,
'''
# coding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #Disable Tensorflow debugging information
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU only='0' ,CPU only='-1'

time_start = time.time()
def Change1to10(y):
    # change the 1dim y data to 10dim
    # y.shape = (m,), y_result.shape = (10,m)
    y_result = np.zeros([10,len(y)])
    for i in range(len(y)):
        y_result[y[i]][i] = 1  # 第x个 -》 第i个的第x个=1
    return y_result


'paras set'
is_train = True  # True:train,False:predict
paras_save_path = sys.path[0] + '/save_data/num_classifier_NN_paras.ckpt'
learn_rate = 0.0001
max_iter = 10000
hide_layer_num_list = [64,9,9,10]  # layer neural num list :like [4,3,3,3] => input dim=4,hide layer = 3 and 3, output dim=3
init_paras_rate = 0.01  # rate of parameters initialization(weight)

'train data create and preprocess'
train_data = load_digits()
# split data
x_train_,x_test_,y_train_,y_test_ = train_test_split(train_data.data,train_data.target,test_size=0.25,random_state=33) # 【：500】
# standard
ss = StandardScaler()
x_train = ss.fit_transform(x_train_)
x_test = ss.fit_transform(x_test_)
# transpose data to shape=(dim,m)
x_train = x_train.transpose()
x_test = x_test.transpose()
y_train = Change1to10(y_train_)
y_test = Change1to10(y_test_)


'init paras'
w = []  # weight list: like w[1] => layer 1 weight,w[0] => input weight[=1]
b = []
w.append(0.)  # w0 not exist,instead of 0
b.append(0.)
for i in range(1,len(hide_layer_num_list)):
    # init weight and b
    w_ = tf.Variable(tf.random_normal([hide_layer_num_list[i],hide_layer_num_list[i-1]],dtype=tf.float32,mean=0.0,stddev=1.0,seed=1)*init_paras_rate)
    w.append(w_)
    b_ = tf.Variable(tf.zeros([hide_layer_num_list[i],1]))
    b.append(b_)
# init train data placeholder
x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
y_score_input = tf.placeholder(tf.float32)  # for accuracy calculation model


'tensorflow model'
# front
z = []  # store z value[ z = w * x + b ]
a = []  # store a value[relu(z)] , a[0] = input x_train
z.append(0.)  # ignore z[0]
a.append(0.)  # ignore a[0]

z.append(tf.matmul(w[1], x) + b[1])  # z[1] = w * x + b
a.append(tf.nn.relu(z[1]))  # a[1]=relu(z[1]) => layer 1 out data
for i in range(2,len(hide_layer_num_list)-1): # 注意此处for次数
    # 第1到n-1层（所有隐层）
    z.append(tf.matmul(w[i], a[i-1]) + b[i])
    a.append(tf.nn.relu(z[i]))
z.append(tf.matmul(w[len(hide_layer_num_list)-1], a[len(hide_layer_num_list)-2]) + b[len(hide_layer_num_list)-1])
# not exist sigmoid function, causing included in loss model


# # loss model 2:(cross_entropy & softmax)
# y = tf.nn.log_softmax(z2)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# fit_model = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

# loss model 1 (sigmoid & cross_entropy)
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=z[len(hide_layer_num_list)-1], labels=y_)  # logits:model predict y \ cross entropy way
fit_model = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

# predict model
y_predict = tf.to_float(tf.argmax(z[len(hide_layer_num_list)-1],axis=0))  # change 10dim to 1dim

# score model
correct_matrix = tf.equal(y_predict,y_score_input)
score_prediction = tf.reduce_mean(tf.cast(correct_matrix,dtype=tf.float32))  # accuracy of input and label

# save paras
saver = tf.train.Saver()

'fit'
with tf.Session() as sess:
    # init Variable
    sess.run(tf.global_variables_initializer())

    # train
    if is_train == True:
        iter_max = 20000
        for i in range(iter_max):
            sess.run(fit_model, feed_dict={x: x_train, y_: y_train})
            accuracy = sess.run(score_prediction, feed_dict={x: x_test, y_score_input: y_test_})
            if i%1000==0 or (i<200 and i%10==0):
                print("accuracy:"+str(accuracy)+"   iter:" + str(i))
        # save paras
        path_save = saver.save(sess, sys.path[0] + '/save_data/' + 'iter_'+ str(iter_max) + 'accuray_' + str(accuracy) + '.ckpt')

    else:
        paras_save_path = './save_data/iter_20000accuray_0.91333336.ckpt'
        saver.restore(sess,paras_save_path)  # not need init variables

    # save paras
    #path_save = saver.save(sess,paras_save_path)
    # print("variables save success:",path_save)

    # predict and accuracy
    print("accuracy:", sess.run(score_prediction, feed_dict={x: x_test, y_score_input: y_test_}))
    print("time used:%f s"%(time.time()-time_start))

    # plt.gray()
    # for i in range(10):
    #     plt.matshow(x_train_[i].reshape(8,8))
    #     plt.title(sess.run(y_predict, feed_dict={x: x_train}))
    # plt.show()
