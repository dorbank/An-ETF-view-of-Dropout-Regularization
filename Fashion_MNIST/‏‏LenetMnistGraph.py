import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
from tensorflow.python.ops.init_ops import random_normal_initializer
from tensorflow.python.ops.gen_nn_ops import relu
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer

CLASSES_NUM = 10
COHERENCE_REGULARIZATION = "None" # choose "None" \ "Convolution" \ FC
coherence_coef = 100 if COHERENCE_REGULARIZATION == "FC" else 1000
weight_decay = 0. #0.005


welchBound = ( (800-400)/( 799.*400. ) )**0.5
print ("welch bound is ", welchBound)


corrMat = tf.fill([800,800], welchBound)
eye = tf.eye(800)
corrMat = eye + (1.0-eye)*corrMat

FC = tf.get_variable("FC", shape=[400,800], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
FC_b = tf.get_variable("b", shape=[800], dtype=tf.float32, initializer=tf.constant_initializer(0.))
x = tf.placeholder(tf.float32, [None, 32, 32, 1])
y = tf.placeholder(tf.int64, [None])
dropout = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(dtype=tf.float32)
weight_name = 0

print ("##################")
print ("##################")
print ("Lenet Mnist Graph")
print ("##################")
print ("##################")
def set_lr(epoch):
    if epoch < 80:
        return 0.001
    if epoch < 110:
        return 0.0005
    if epoch < 150:
        return 0.0001
    if epoch < 180:
        return 0.00005
    return 0.00001

def max_pool(input, height, width, depth = 1):
    input = tf.nn.max_pool(input, ksize=[1, height, width, depth], strides=[1, height, width, depth], padding="SAME")
    print ("batch shape after "+str(height)+"x"+str(width)+" max_pool:\t", input.shape)
    return input

def conv2d(input,height,width, outChannels, stride=1):
    global weight_name
    W_patch = tf.get_variable(name="W_patch"+str(weight_name), shape=[height,width,input.shape[3],outChannels], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b_patch = tf.get_variable(name="b_patch" + str(weight_name), shape=[outChannels],initializer=tf.constant_initializer(0.))
    weight_name += 1
    input = tf.nn.conv2d(input, filter=W_patch, strides=[1, stride, stride, 1],padding="VALID")
    input = tf.nn.bias_add(input, b_patch)
    input = tf.nn.relu(input)
    print("batch shape after "+str(height)+"x"+str(width)+" conv:\t", input.shape)
    return input

def conv2d_co(input,height,width,outChannels, stride=1):
    global weight_name
    W_patch = tf.get_variable(name="W_patch"+str(weight_name), shape=[height,width,input.shape[3],outChannels], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b_patch = tf.get_variable(name="b_patch" + str(weight_name), shape=[outChannels],initializer=tf.constant_initializer(0.))
    W_mat = tf.reshape(W_patch, [height * width * input.shape[3], outChannels])
    weight_name += 1
    input = tf.nn.conv2d(input, filter=W_patch, strides=[1, stride, stride, 1],padding="VALID")
    input = tf.nn.bias_add(input, b_patch)
    input = tf.nn.relu(input)
    print("batch shape after "+str(height)+"x"+str(width)+" conv:\t", input.shape)

    eye = tf.eye(outChannels)
    weightCoherenceLoss = tf.reduce_max(tf.abs(tf.matmul(W_mat, W_mat, transpose_a=True)) - eye)
    coherence_loss = coherence_coef*weightCoherenceLoss
    return input, coherence_loss

z = conv2d(x,5,5,6)
z = max_pool(z,2,2)
if COHERENCE_REGULARIZATION == "Convolution":
    z, coherence_loss = conv2d_co(z, 5, 5, 16)
    z = max_pool(z,2,2)
    z = tf.reshape(z,[-1,z.shape[1]*z.shape[2]*z.shape[3]])
    z = tf.nn.dropout(z,keep_prob=dropout)

else:
    z = conv2d(z, 5, 5, 16)
    z = max_pool(z,2,2)
    z = tf.reshape(z,[-1,z.shape[1]*z.shape[2]*z.shape[3]])

layer = Dense(800, activation=relu,kernel_initializer=xavier_initializer(uniform=False), name="Dense")
z = layer.apply(z)
layer.kernel = tf.nn.l2_normalize(layer.kernel, axis=0)
eye = tf.eye(800)

if COHERENCE_REGULARIZATION == "FC":
    coherence_loss = coherence_coef * tf.reduce_max(tf.abs(tf.abs(tf.matmul(layer.kernel, layer.kernel, transpose_a=True))-corrMat))
else:
    coherence_loss = 0
coherence = tf.reduce_max(tf.abs(tf.abs(tf.matmul(layer.kernel, layer.kernel, transpose_a=True))-eye))
z = tf.nn.dropout(z,keep_prob=dropout)
pred_logits = tf.contrib.layers.fully_connected(z,CLASSES_NUM,activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
print ("z.shape is ", pred_logits.shape)
loss = tf.losses.sparse_softmax_cross_entropy(logits=pred_logits, labels=y, reduction=tf.losses.Reduction.MEAN)
l2_loss = tf.reduce_sum([tf.nn.l2_loss(i) for i in tf.trainable_variables()])
cost = loss  + weight_decay*l2_loss + coherence_loss
pred = tf.argmax(pred_logits, 1)
correct_pred = tf.equal(pred, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
