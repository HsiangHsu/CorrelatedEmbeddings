import tensorflow as tf
import numpy as np

_IMAGE_SIZE = 32
_IMAGE_CHANNELS = 3
_NUM_CLASSES = 100
_RESHAPE_SIZE = 4*4*128
d = _NUM_CLASSES-1
epsilon = 1e-2
alpha = 0.1

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

def lrelu(x, alpha=0.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def model1(x):
    with tf.name_scope('data'):
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

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
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv1', conv1)
    tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv2', conv2)
    tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv3', conv3)
    tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

    with tf.variable_scope('conv4') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv4', conv4)
    tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))

    with tf.variable_scope('conv5') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv5', conv5)
    tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('fully_connected1') as scope:
        reshape = tf.reshape(pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    with tf.variable_scope('fully_connected2') as scope:
        weights = variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc2', local4)
    tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

    with tf.variable_scope('output') as scope:
        weights = variable_with_weight_decay('weights', [192, d], stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases', [d], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/output', softmax_linear)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    # y_pred_cls = tf.argmax(softmax_linear, axis=1)

    constF = tf.fill([tf.shape(softmax_linear)[0],1],np.float32(1))
    final_output_F = tf.concat([constF,softmax_linear],axis=1)
    
    return final_output_F, global_step
    #return x, y, final_output_F, global_step, y_pred_cls

def model2(x):
    with tf.name_scope('data'):
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

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

    with tf.variable_scope('conv12') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv1', conv1)
    tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv22') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv2', conv2)
    tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv32') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv3', conv3)
    tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

    with tf.variable_scope('conv42') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv4', conv4)
    tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))

    with tf.variable_scope('conv52') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv5', conv5)
    tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('fully_connected12') as scope:
        reshape = tf.reshape(pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    with tf.variable_scope('fully_connected22') as scope:
        weights = variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc2', local4)
    tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

    with tf.variable_scope('output2') as scope:
        weights = variable_with_weight_decay('weights', [192, d], stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases', [d], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/output', softmax_linear)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    # y_pred_cls = tf.argmax(softmax_linear, axis=1)

    constG = tf.fill([tf.shape(softmax_linear)[0],1],np.float32(1))
    final_output_G = tf.concat([constG,softmax_linear],axis=1)
    
    return final_output_G, global_step

# G Net
# Layer 1
G_W1 = tf.Variable(xavier_init([_NUM_CLASSES, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

# Layer 2
G_W2 = tf.Variable(xavier_init([128, 64]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[64]), name='G_b2')

# Layer 3
G_W3 = tf.Variable(xavier_init([64, d]), name='G_W3')
G_b3 = tf.Variable(tf.zeros(shape=[d]), name='G_b3')

def G_net(y):
    G_h1 = tf.nn.relu(tf.matmul(y, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_logit = tf.matmul(G_h2, G_W3) + G_b3
    
    constG = tf.fill([tf.shape(G_logit)[0],1],np.float32(1))
    final_output_G = tf.concat([constG,G_logit],axis=1)

    z = tf.eye(_NUM_CLASSES)
    G_h12 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h22 = tf.nn.relu(tf.matmul(G_h12, G_W2) + G_b2)
    G_logit2 = tf.matmul(G_h22, G_W3) + G_b3
    
    constG2 = tf.fill([tf.shape(G_logit2)[0],1],np.float32(1))
    output_G_all_labels = tf.concat([constG2,G_logit2],axis=1)
    return final_output_G, output_G_all_labels

def F_loss_svd(f_out, g_out, mb_size):
    """
    Create the loss function that will be minimized by the fg-net. Many options exist.
    The implementation below uses the 1-Schatten norm from the derivation. It might slow.
    
    
    Inputs: f_out and g_out, which are tensors of the same shape produced by the outut
            of the f and g nets. Assumes that they are of the form (#batches, #output).
            
    Outputs: returns objective
    """
    clip_min = np.float32(-10000)
    clip_max=np.float32(10000)
    # number of samples in batch
    nBatch = mb_size
    
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
    invCorrF = tf.matrix_inverse(corrF+epsilon*tf.eye(n), adjoint=True)
    
    prodGiFG = tf.matmul(tf.matmul(tf.transpose(corrFG),invCorrF),corrFG)
    
    s,v = tf.self_adjoint_eig(prodGiFG)
    
    schatNorm = tf.reduce_sum(tf.sqrt(tf.abs(s)))
    
    # define objective
    objective = sqG - 2*schatNorm 
    
    #return objective
    return objective, schatNorm
