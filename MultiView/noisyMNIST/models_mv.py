import tensorflow as tf
import numpy as np
import scipy as sp

# d = 9
epsilon = 1e-2

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def F_net(x, d):
    F_W1 = tf.Variable(xavier_init([5, 5, 1, 32]), name='F_W1')
    F_b1 = tf.Variable(tf.zeros(shape=[32]), name='F_b1')

    # Layer 2
    F_W2 = tf.Variable(xavier_init([5, 5, 32, 64]), name='F_W2')
    F_b2 = tf.Variable(tf.zeros(shape=[64]), name='F_b2')

    # Layer 3
    F_W3 = tf.Variable(xavier_init([7 * 7 * 64, 1024]), name='F_W3')
    F_b3 = tf.Variable(tf.zeros(shape=[1024]), name='F_b3')

    # Layer 4
    F_W4 = tf.Variable(xavier_init([1024, 1024]), name='F_W4')
    F_b4 = tf.Variable(tf.zeros(shape=[1024]), name='F_b4')

    # Layer 5
    F_W5 = tf.Variable(xavier_init([1024, d]), name='F_W5')
    F_b5 = tf.Variable(tf.zeros(shape=[d]), name='F_b5')
    
    # First Layer
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    F_h1 = tf.nn.relu(conv2d(x_image, F_W1) + F_b1)  
    F_pool1 = max_pool_2x2(F_h1)  
                      
    # Second Layer
    F_h2 = tf.nn.relu(conv2d(F_pool1, F_W2) + F_b2)
    F_pool2 = max_pool_2x2(F_h2)
    
    # Fully Connected Layer
    F_pool2_flat = tf.reshape(F_pool2, [-1, 7*7*64])
    F_fc1 = tf.nn.relu(tf.matmul(F_pool2_flat, F_W3) + F_b3)

    #
    F_fc2 = tf.nn.relu(tf.matmul(F_fc1, F_W4) + F_b4)
    
    # Readout Layer and Normalization                  
    F_logit = tf.matmul(F_fc2, F_W5) + F_b5
    
    constF = tf.fill([tf.shape(F_logit)[0],1],np.float32(1))
    final_output_F = tf.concat([constF,F_logit],axis=1)
    return final_output_F

def G_net(y, d):
    G_W1 = tf.Variable(xavier_init([5, 5, 1, 32]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[32]), name='G_b1')

    # Layer 2
    G_W2 = tf.Variable(xavier_init([5, 5, 32, 64]), name='G_W2')
    G_b2 = tf.Variable(tf.zeros(shape=[64]), name='G_b2')

    # Layer 3
    G_W3 = tf.Variable(xavier_init([7 * 7 * 64, 1024]), name='G_W3')
    G_b3 = tf.Variable(tf.zeros(shape=[1024]), name='G_b3')

    # Layer 4
    G_W4 = tf.Variable(xavier_init([1024, 1024]), name='G_W4')
    G_b4 = tf.Variable(tf.zeros(shape=[1024]), name='G_b4')

    # Layer 5
    G_W5 = tf.Variable(xavier_init([1024, d]), name='G_W5')
    G_b5 = tf.Variable(tf.zeros(shape=[d]), name='G_b5')

    y_image = tf.reshape(y, [-1, 28, 28, 1])
    G_h1 = tf.nn.relu(conv2d(y_image, G_W1) + G_b1)  
    G_pool1 = max_pool_2x2(G_h1)

    # Second Layer
    G_h2 = tf.nn.relu(conv2d(G_pool1, G_W2) + G_b2)
    G_pool2 = max_pool_2x2(G_h2)
    
    # Fully Connected Layer
    G_pool2_flat = tf.reshape(G_pool2, [-1, 7*7*64])
    G_fc1 = tf.nn.relu(tf.matmul(G_pool2_flat, G_W3) + G_b3)

    G_fc2 = tf.nn.relu(tf.matmul(G_fc1, G_W4) + G_b4)
    
    # Readout Layer and Normalization                  
    G_logit = tf.matmul(G_fc1, G_W5) + G_b5
    
    constG = tf.fill([tf.shape(G_logit)[0],1],np.float32(1))
    final_output_G = tf.concat([constG,G_logit],axis=1)
    return final_output_G

def F_loss_svd(f_out, g_out, mb_size):
    """
    Create the loss function that will be minimized by the fg-net. Many options exist.
    The implementation below uses the 1-Schatten norm from the derivation. It might slow.
    
    
    Inputs: f_out and g_out, which are tensors of the same shape produced by the outut
            of the f and g nets. Assumes that they are of the form (#batches, #output).
            
    Outputs: returns objective
    """
    th = 10000
    clip_min = np.float32(-th)
    clip_max=np.float32(th)
    # number of samples in batch
    nBatch = mb_size
    
    # we clip f to avoid runaway arguments
    # f_clip = tf.clip_by_value(f_out,clip_min,clip_max)
    # g_clip = tf.clip_by_value(g_out,clip_min,clip_max)
    f_clip = f_out
    g_clip = f_out
    
    # create correlation matrices
    corrF = tf.matmul(tf.transpose(f_clip),f_clip)/nBatch
    corrFG = tf.matmul(tf.transpose(f_clip),g_clip)/nBatch
    
    # Second moment of g
    sqG = tf.reduce_sum(tf.reduce_mean(tf.square(g_clip),0))
    
    # compute svd in objective
    n = tf.shape(corrF)[0] 
    
    #correction term
    epsilon =1e-3
    
    invCorrF = tf.matrix_inverse(corrF, adjoint=True) # + epsilon*tf.eye(n)
    
    prodGiFG = tf.matmul(tf.matmul(tf.transpose(corrFG),invCorrF),corrFG)
    
    s,v = tf.self_adjoint_eig(prodGiFG)
    
    schatNorm = tf.reduce_sum(tf.sqrt(tf.abs(s)))
    
    # define objective
    objective = sqG - 2*schatNorm 
    
    #return objective
    return objective, schatNorm

def normalizeFG(F,G):
    # Values for G
    Gs = G[:,1:]
    b_mean = Gs.mean(axis=0)
    Gs = Gs - b_mean
    corrG = Gs.transpose().dot(Gs)/Gs.shape[0]
    U,v,_ = np.linalg.svd(corrG)
    corrG_sqrt_inv = (U*(v)**(-.5)).dot(U.transpose())
    
    b_mean = np.concatenate(([0],b_mean))
    B = sp.linalg.block_diag(1,corrG_sqrt_inv)
    
    nG = (G-b_mean).dot(B)

    # values for F
    Fs = F[:,1:]
    a_mean = Fs.mean(axis=0)
    Fs = Fs - a_mean
    corrF = Fs.transpose().dot(Fs)/Fs.shape[0]
    U,v,_ = np.linalg.svd(corrF)
    corrF_sqrt_inv = (U*(v)**(-.5)).dot(U.transpose())
    

    a_mean = np.concatenate(([0],a_mean))
    A = sp.linalg.block_diag(1,corrF_sqrt_inv)
    
    nF = (F-a_mean).dot(A)
    
    # Create proper normalization
    U,s,V = np.linalg.svd(nF.transpose().dot(nG)/G.shape[0])

    return A.dot(U),a_mean,B.dot(V.transpose()),b_mean

def normalization(F, G, A, B, a, b):
    wF = (F-a).dot(A)
    wG = (G-b).dot(B)
    return wF, wG

def H_net(x):
    # First layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout to avoid overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

    # Readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

def loss_P_YgX(f_out, g_out, mb_size, likelihood):

    # define clip ranges
    clip_min = np.float32(-10000)
    clip_max=np.float32(10000)
    # number of samples in batch
    nBatch = mb_size
    
    # we clip f to avoid runaway arguments
    f_clip = tf.clip_by_value(f_out,clip_min,clip_max)

    # C_f
    corrF = tf.matmul(tf.transpose(f_clip),f_clip)/nBatch

    # C_{fg}
    corrFG = tf.matmul(tf.transpose(f_clip), tf.matmul(likelihood, g_out))/nBatch

    # Second moment of g
    sqG = tf.reduce_sum(tf.matmul(likelihood, tf.reshape(tf.reduce_sum(tf.square(g_out), axis=1), [10, 1])))/nBatch

    # compute svd in objective
    n = tf.shape(corrF)[0] 
    
    #correction term
    epsilon = 1e-3
    
    invCorrF = tf.matrix_inverse(corrF+epsilon*tf.eye(n), adjoint=True) #check
    # print(invCorrF.shape)
    
    prodGiFG = tf.matmul(tf.matmul(tf.transpose(corrFG),invCorrF),corrFG)
    
    s,v = tf.self_adjoint_eig(prodGiFG)
    
    schatNorm = tf.reduce_sum(tf.sqrt(tf.abs(s)))
    
    # define objective
    objective = sqG - 2*schatNorm 
    
    return objective, schatNorm
