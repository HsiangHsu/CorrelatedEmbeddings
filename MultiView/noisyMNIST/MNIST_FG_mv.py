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

# from util import load_data
from models_mv import *

# Initialize the input of F and G nets
X = tf.placeholder(tf.float32, shape=[None, 784], name='X1')
Y = tf.placeholder(tf.float32, shape=[None, 784], name='X2')

with open ('noisy_mnist', 'rb') as f1:
    n_mnist = pickle.load(f1)

with open ('noisy_rotate_mnist', 'rb') as f2:
    nr_mnist = pickle.load(f2)

train_x = n_mnist[0][0]
train_y = nr_mnist[0][0]
test_x = n_mnist[2][0]
test_y = nr_mnist[2][0]

train_labels = n_mnist[0][1].reshape((len(n_mnist[0][1]), 1))
test_labels = n_mnist[2][1].reshape((len(n_mnist[2][1]), 1))


def train(var_name, mb_size, n_steps, d):
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Construct networks
    F_Net = F_net(X, d)
    G_Net= G_net(Y, d)

    # Loss functions
    F_loss, _ = F_loss_svd(G_Net, F_Net, mb_size)
    G_loss = F_loss

    # Gradient-based solver
    learning_rate = 0.01
    F_solver = tf.train.AdagradOptimizer(learning_rate).minimize(F_loss)
    G_solver = tf.train.AdagradOptimizer(learning_rate).minimize(G_loss)

    # run model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()

    file = open('Data/'+var_name+'_log.txt','w')
    file.write('Iter\t sec/iter \t total sec\n')
    t0 = time.time()
    for it in range(n_steps):
        randidx = np.random.randint(len(train_x), size=mb_size)
        X_mb = train_x[randidx]
        Y_mb = train_y[randidx]

        _, F_loss_curr = sess.run([F_solver, F_loss], feed_dict={X: X_mb, Y: Y_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb, Y: Y_mb})
        if (it+1)%1000 == 0:
                t1 =time.time()
                total_time = t1-t0
                time_per_iter = total_time/(it+1)
                file.write('{:.0f}\t {:.5f} \t {:.5f}\n'.format(it+1,time_per_iter,total_time))
                file.flush()

    # evaluate on train set
    print('Reconstruct the results')
    F_output_train = sess.run(F_Net, feed_dict={X: train_x})
    G_output_train = sess.run(G_Net, feed_dict={Y: train_y})

    # A, a, B, b = normalizeFG(F_output_train, G_output_train)
    # wF_train, wG_train = normalization(F_output_train, G_output_train, A, B, a, b)

    # trueCorr_train, corrG_train, newCorr_train, Anorm_train, Bnorm_train, wF_train, wG_train = computeMetrics(F_output_train, G_output_train)

    # mnist_train_labels = np.zeros((len(F_output_train), 1))
    # for k in range(len(F_output_train)):
    #     mnist_train_labels[k] = list(mnist.train.labels[k]).index(1)
    # print(F_output_train.shape)
    # print(G_output_train.shape)
    # print(len(train_labels))
    df_data_train = pd.DataFrame(np.hstack((F_output_train, G_output_train, train_labels)))
    df_data_train.to_csv('Data_FG_new/'+var_name+'_train.csv')

    # evaluate on train set
    F_output_test = sess.run(F_Net, feed_dict={X: test_x})
    G_output_test = sess.run(G_Net, feed_dict={Y: test_y})

    # wF_test, wG_test = normalization(F_output_test, G_output_test, A, B, a, b)
    # trueCorr_test, corrG_test, newCorr_test, Anorm_test, Bnorm_test, wF_test, wG_test = computeMetrics(F_output_test, G_output_test)

    # mnist_test_labels = np.zeros((len(F_output_test), 1))
    # for k in range(len(F_output_test)):
    #    mnist_test_labels[k] = list(mnist.test.labels[k]).index(1)

    df_data_test = pd.DataFrame(np.hstack((F_output_test, G_output_test, test_labels)))
    df_data_test.to_csv('Data_FG_new/'+var_name+'_test.csv')

    # save_path = saver.save(sess, var_name+'.ckpt')

    file.close()
    sess.close()
        
    
## creates and saves tensorflow models for each of the digits
if __name__ == '__main__':
    var_name = 'MNIST_FG_mv'
    d_list = [40]
    for i in range(len(d_list)):
        print(i)
        train(var_name+str(d_list[i]), 2048, 50000, d_list[i])
