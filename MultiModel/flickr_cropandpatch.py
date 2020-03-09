# -*- coding: utf-8 -*-
""""
Codes for JSAIT on Multi-Modal Learning: Image Tagging Experiments
Author: Hsiang Hsu
email:  hsianghsu@g.harvard.edu
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
import sys

# filenames = os.listdir('flickr30k_images/flickr30k_images')

with open('flickr_filenames.pickle', "rb") as input_file:
    filenames = pickle.load(input_file)['filenames']

image_height = 64
image_width = 64

n = 30000
m = len(filenames)-n
batchsize = 200
number_of_batch = int(n/batchsize)

k = int(sys.argv[1])
# resized the images to
sess = tf.Session()

# for i in range(number_of_batch):
for i in range(k, k+1):
    images_data = np.zeros((batchsize, image_height, image_width, 3))
    img_names = []
    for j in range(batchsize):
        img= mpimg.imread('flickr30k_images/flickr30k_images/'+filenames[i*batchsize+j])/255.0
        images_data[j, :, :, :] = tf.compat.v1.image.resize_images(img, (image_height, image_width)).eval(session=sess)
        img_names.append(filenames[i*batchsize+j])

    save_img = {}
    save_img['images'] = images_data
    save_img['img_names'] = img_names
    pickle_save_file = 'pickle_data/flickr_images'+str(i)+'.pickle'
    f = open(pickle_save_file, 'wb')
    pickle.dump(save_img, f, 2)
    f.close()

# filenames = os.listdir('flickr30k_images/flickr30k_images')
#
# image_height = 218
# image_width = 178
# n = len(filenames)
#
# # resized the images to
# sess = tf.Session()
# i = 0
# for filename in filenames:
#     img = mpimg.imread('flickr30k_images/flickr30k_images/'+filename)/255.0
#     resized_img = tf.compat.v1.image.resize_images(img, (image_height, image_width))
#     resized_img = resized_img.eval(session=sess)
#     mpimg.imsave('flickr30k_images/resized_flickr30k_images/'+filename, resized_img, format='jpg')
#     print(str(i)+' / '+str(n))
#     i = i + 1
