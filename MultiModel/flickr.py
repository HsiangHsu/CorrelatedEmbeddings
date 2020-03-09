# -*- coding: utf-8 -*-
""""
Codes for JSAIT on Multi-Modal Learning: Image Tagging Experiments
Author: Hsiang Hsu
email:  hsianghsu@g.harvard.edu
"""
import tensorflow as tf
import pickle
import gzip
import numpy as np
from time import localtime, strftime
import time
import os

from util import *

# load data
# with open('imagename_embedding.pickle', "rb") as input_file:
#     imagename_embedding = pickle.load(input_file)
#
filenames = os.listdir('pickle_data')

n_batch = len(filenames)
# n_batch = 50

with open('X_Y_500.pickle', "rb") as input_file:
    X_Y = pickle.load(input_file)
Y_data = X_Y['Y']

n_train = 30000
batch_size = 200
n_iteration = 100
learning_rate = 5e-5

image_height = 64
image_width = 64
image_channels = 3



dx = image_height*image_width*image_channels
# dy = 4874
dy = 500
d = 50
epsilon = 1e-3
num_neuron = 100

def train(var_name, mb_size = 128, n_steps = 10, lr=1e-4):
    file = open(var_name+'_log.txt','w')
    file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
    file.write('=== Dataset Summary ===\n')
    file.write('Training set: {}, {}\n'.format((30000, 218, 178, 3), (30000, 4874)))
    file.flush()

    file.write('=== Parameter Summary ===\n')
    file.write('dx = {}, dy = {}, n_train = {}\n'.format(dx, dy, n_train))
    file.write('Iteration: {}, learning_rate: {}, batch_size: {}\n'.format(n_steps, lr, mb_size))
    file.write('number of batch: {}\n'.format(n_batch))
    file.flush()

    file.write('Initializing Placeholders\n')
    file.flush()
    X = tf.placeholder(tf.float32, [None, image_height, image_width, image_channels])
    Y = tf.placeholder(tf.float32, [None, dy])

    file.write('Initializing Neural Networks\n')
    file.flush()
    # Construct networks
    f_output = F_net(X, d, image_height, image_channels)
    g_output = G_net(Y, d, dy, num_neuron)

    # Loss functions
    objective, _ = create_loss_svd(f_output, g_output)
    step = tf.train.GradientDescentOptimizer(lr).minimize(objective)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    file.write('Start Training\n')
    file.flush()
    file.write('Iter\t batch\t Loss\t total sec\n')
    t0 = time.time()
    for it in range(n_steps):
        for m in range(n_batch):
            with open('pickle_data/flickr_images'+str(m)+'.pickle', "rb") as input_file:
                x_mb = pickle.load(input_file)['images']
            y_mb = Y_data[m*mb_size:m*mb_size+mb_size]
            # file.write('{}, {}\n'.format(x_mb.shape, y_mb.shape))
            file.flush()

            sess.run(step, feed_dict= {X: x_mb, Y: y_mb})
            # _, current_loss = sess.run([solver_psgx, cross_entropy], feed_dict={X: x_mb, S: s_mb})

            if m % 10 == 0:
                t1 =time.time()
                total_time = t1-t0
                current_loss = objective.eval(feed_dict={X: x_mb, Y: y_mb}, session=sess)
                file.write('{}\t {}\t {:.4f}\t {:.4f}\n'.format(it, m, current_loss, total_time))
                file.flush()

            del x_mb
            del y_mb

    file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
    file.flush()

    # evaluate on train set
    file.write('Reconstruct the results\n')
    file.flush()

    f_output_all = np.zeros((n_batch*mb_size, d+1))
    g_output_all = np.zeros((n_batch*mb_size, d+1))
    for m in range(n_batch):
        with open('pickle_data/flickr_images'+str(m)+'.pickle', "rb") as input_file:
            x_mb = pickle.load(input_file)['images']
        y_mb = Y_data[m*mb_size:m*mb_size+mb_size]

        f_output_batch = sess.run(f_output, feed_dict={X: x_mb})
        g_output_batch = sess.run(g_output, feed_dict={Y: y_mb})

        f_output_all[m*mb_size:m*mb_size+mb_size, :] = f_output_batch
        g_output_all[m*mb_size:m*mb_size+mb_size, :] = g_output_batch

        del x_mb
        del y_mb
    # f_all, g_all = whiten(f_output_all, g_output_all, f_output_all, g_output_all)

    # f_ = sess.run(F_Net, feed_dict= {X: np.identity(Y_data.shape[1])})
    A, a, B, b = normalizeFG(f_output_all, g_output_all)
    f_all, g_all = (f_output_all-a).dot(A), (g_output_all-b).dot(B)

    g_ = sess.run(g_output, feed_dict= {Y: np.identity(Y_data.shape[1])})
    g_tokens = (g_-b).dot(B)


    pickle_save_file = 'data/'+var_name+'_FG.pickle'
    f = open(pickle_save_file, 'wb')
    save = {
        'f_all': f_all,
        'g_all': g_all,
        'g_tokens': g_tokens
        }
    pickle.dump(save, f, 2)
    f.close()

    file.write('finished!!!\n')
    file.flush()


    sess.close()


## creates and saves tensorflow models for each of the digits
if __name__ == '__main__':
    var_name = 'flickr'+'_'+strftime("%Y-%m-%d-%H.%M.%S", localtime())
    train(var_name, mb_size = batch_size, n_steps = n_iteration, lr=learning_rate)
