# -*- coding: utf-8 -*-
""""
Convoutional NN script for MNIST based on Tensorflow Tutorial.

Author: fdcalmon@us.ibm.com
"""

import tensorflow as tf
import pickle
import gzip
import numpy as np
import scipy as sp
import time
import pandas as pd
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
# import tensorflow as tf
from sklearn.metrics import confusion_matrix
# from time import time

from data import *
from model import *

# train_x, train_y, train_l, train_l_fine = get_data_set("train", cifar=10)
# test_x, test_y, test_l, test_l_fine = get_data_set("test", cifar=10)

train_x, train_y, train_l = get_data_set("train", cifar=10)
test_x, test_y, test_l = get_data_set("test", cifar=10)



_IMAGE_SIZE = 32
_IMAGE_CHANNELS = 3
_BATCH_SIZE = 256
_NUM_CLASSES = 100
_ITERATION = 30000
# _SAVE_PATH = "./tensorboard/cifar-10/"

X = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS])
Y = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS])

def train(var_name, mb_size, n_steps):
    F_Net, global_step1 = model1(X)
    G_Net, global_step2 = model2(Y)   

    # Loss functions
    F_loss, prodGiFG = F_loss_svd(F_Net, G_Net, mb_size)
    G_loss = F_loss

    # Gradient-based solver
    learning_rate = 0.01
    # F_solver = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(F_loss, global_step=global_step)
    F_solver = tf.train.AdagradOptimizer(learning_rate).minimize(F_loss)
    G_solver = tf.train.AdagradOptimizer(learning_rate).minimize(G_loss)

    # run model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    file = open('Data/'+var_name+'_log.txt','w')

    # Part 2
    # file.write('=== Part 2 ===\n')
    file.write('Iter\t sec/iter \t total sec\n')
    t0 = time.time()
    for it in range(n_steps):
        # print(it)
        c = np.random.randint(10)
        class_X = train_x[train_l==c]
        randidx1 = np.random.randint(len(class_X), size=_BATCH_SIZE)
        randidx2 = np.random.randint(len(class_X), size=_BATCH_SIZE)
        X_mb = class_X[randidx1]
        Y_mb = class_X[randidx2]

        _, F_loss_curr = sess.run([F_solver, F_loss], feed_dict={X: X_mb, Y: Y_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb, Y: Y_mb})
        print(F_loss_curr)
        if (it+1)%1000 == 0:
                t1 =time.time()
                total_time = t1-t0
                time_per_iter = total_time/(it+1)
                file.write('{:.0f}\t {:.5f} \t {:.5f}\n'.format(it+1,time_per_iter,total_time))
                file.flush()

    # evaluate on train set
    print('Reconstruct the results')
    #train_labels = np.zeros((50000, 1))
    #train_labels_fine = np.zeros((50000, 1))
    #for k in range(50000):
    #    train_labels[k] = list(train_y[k, :]).index(1)
    #    train_labels_fine[k] = list(train_y[k, :]).index(1)

    F_output_train = sess.run(F_Net, feed_dict={X: train_x[0:10000]})
    G_output_train = sess.run(G_Net, feed_dict={Y: train_x[0:10000]})
    # print(train_x[0:5])
    data_train = np.hstack((F_output_train, G_output_train, np.array(train_l[0:10000]).reshape((10000, 1))))
    
    train_sample_split = [10000, 20000, 30000, 40000, 50000]
    for i in range(len(train_sample_split)-1):
        F_output_train = sess.run(F_Net, feed_dict={X: train_x[train_sample_split[i]:train_sample_split[i+1]]})
        G_output_train = sess.run(G_Net, feed_dict={Y: train_x[train_sample_split[i]:train_sample_split[i+1]]})
        data_train_temp = np.hstack((F_output_train, G_output_train, np.array(train_l[train_sample_split[i]:train_sample_split[i+1]]).reshape((10000, 1))))
        data_train = np.vstack((data_train, data_train_temp))
    # F_output_train = sess.run(output, feed_dict={x: train_x})
    # G_output_train = sess.run(G_Net, feed_dict={y: train_y})
    
    # data_train = np.hstack((F_output_train, G_output_train, np.array(train_l).reshape((50000, 1))))
    df_data_train = pd.DataFrame(data_train)
    df_data_train.to_csv('Data_FG/'+var_name+'_train.csv')

    # evaluate on test set
    F_output_test = sess.run(F_Net, feed_dict={X: test_x})
    G_output_test = sess.run(G_Net, feed_dict={Y: test_x})

    #test_labels = np.zeros((len(F_output_test), 1))
    #for k in range(len(F_output_test)):
    #    test_labels[k] = list(test_y[k, :]).index(1)

    df_data_test = pd.DataFrame(np.hstack((F_output_test, G_output_test, np.array(test_l).reshape((10000, 1)))))
    df_data_test.to_csv('Data_FG/'+var_name+'_test.csv')


    file.close()
    sess.close()
        
    
## creates and saves tensorflow models for each of the digits
if __name__ == '__main__':
    var_name = 'cifar100'
    train(var_name, mb_size = _BATCH_SIZE, n_steps = _ITERATION)
