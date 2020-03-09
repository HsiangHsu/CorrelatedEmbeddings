import tensorflow as tf
import numpy as np
import scipy as sp


def create_loss_svd(f_out,g_out,clip_min = np.float32(-10000),clip_max=np.float32(10000)):
    """
    Create the loss function that will be minimized by the fg-net. Many options exist.
    The implementation below uses the 1-Schatten norm from the derivation. It might slow.


    Inputs: f_out and g_out, which are tensors of the same shape produced by the outut
            of the f and g nets. Assumes that they are of the form (#batches, #output).

    Outputs: returns objective
    """

    # number of samples in batch
    nBatch = tf.cast(tf.shape(f_out)[0],tf.float32)

    # we clip f to avoid runaway arguments
    # f_clip = tf.clip_by_value(f_out,clip_min,clip_max)
    f_clip = f_out

    # create correlation matrices
    corrF = tf.matmul(tf.transpose(f_clip),f_clip)/nBatch
    corrFG = tf.matmul(tf.transpose(f_clip),g_out)/nBatch

    # Second moment of g
    sqG = tf.reduce_sum(tf.reduce_mean(tf.square(g_out),0))

    # compute svd in objective
    n = tf.shape(corrF)[0]

    #correction term
    epsilon =1e-4

    invCorrF = tf.matrix_inverse(corrF+epsilon*tf.eye(n), adjoint=True) #check


    prodGiFG = tf.matmul(tf.matmul(tf.transpose(corrFG),invCorrF),corrFG)

    s,v = tf.self_adjoint_eig(prodGiFG)
    #s,u,v = tf.svd(prodGiFG)

    schatNorm = tf.reduce_sum(tf.sqrt(tf.abs(s)))


    # define objective
    objective = sqG - 2*schatNorm #+ tf.trace(corrF)#.3*tf.reduce_sum(tf.square((corrF-tf.eye(n))))

    #return objective
    return objective,schatNorm

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

def whiten(F, G, Fb, Gb):
    A, a, B, b = normalizeFG(Fb, Gb)
    return (F-a).dot(A), (G-b).dot(B)

def F_net(x, d, _IMAGE_SIZE, _IMAGE_CHANNELS):
    _RESHAPE_SIZE = 4*4*512

    # with tf.name_scope('data'):
        # x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        # y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        # x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')


    def variable_with_weight_decay(name, shape, stddev, wd):
        dtype = tf.float32
        var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv1', conv1)
    # tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv2', conv2)
    # tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv3', conv3)
    # tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

    with tf.variable_scope('conv4') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv4', conv4)
    # tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))

    with tf.variable_scope('conv5') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv5', conv5)
    # tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm5 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool5 = tf.nn.max_pool(norm5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('fully_connected1') as scope:
        reshape = tf.reshape(pool5, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.tanh(tf.matmul(reshape, weights) + biases, name=scope.name)
    # tf.summary.histogram('Fully connected layers/fc1', local3)
    # tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    with tf.variable_scope('fully_connected2') as scope:
        weights = variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.tanh(tf.matmul(local3, weights) + biases, name=scope.name)
    # tf.summary.histogram('Fully connected layers/fc2', local4)
    # tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

    with tf.variable_scope('output') as scope:
        weights = variable_with_weight_decay('weights', [192, d], stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases', [d], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    # tf.summary.histogram('Fully connected layers/output', softmax_linear)
    constF = tf.fill([tf.shape(softmax_linear)[0],1],np.float32(1))
    final_output_F = tf.concat([constF, softmax_linear],axis=1)
    return final_output_F

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def G_net(y, d, dy, num_neuron):
    G_W1 = tf.Variable(xavier_init([dy, num_neuron]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[num_neuron]), name='G_b1')
    G_W2 = tf.Variable(xavier_init([num_neuron, 64]), name='G_W2')
    G_b2 = tf.Variable(tf.zeros(shape=[64]), name='G_b2')
    G_W3 = tf.Variable(xavier_init([64, 32]), name='G_W3')
    G_b3 = tf.Variable(tf.zeros(shape=[32]), name='G_b3')
    G_W4 = tf.Variable(xavier_init([32, d]), name='G_W4')
    G_b4 = tf.Variable(tf.zeros(shape=[d]), name='G_b4')

    fc1 = tf.nn.relu(tf.matmul(y, G_W1) + G_b1)
    fc2 = tf.nn.relu(tf.matmul(fc1, G_W2) + G_b2)
    fc3 = tf.nn.relu(tf.matmul(fc2, G_W3) + G_b3)
    g = tf.matmul(fc3, G_W4) + G_b4

    constG = tf.fill([tf.shape(g)[0],1],np.float32(1))
    final_output_G = tf.concat([constG, g],axis=1)
    return final_output_G
