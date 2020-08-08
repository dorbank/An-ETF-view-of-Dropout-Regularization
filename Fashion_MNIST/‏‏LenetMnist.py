import tensorflow as tf
import mnist_reader
import math
import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import LenetMnistGraph as graph



GPU = 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU)
#os.environ["CUDA_VISIBLE_DEVICES"]= ''
SAVE = True #true if we save the models
LOAD = False #true if we load a model
KEEP_TRAINING = True


LOAD_DIRECTORY_NAME = "saved_Lenet"
SAVE_DIRECTORY_NAME = "saved_Lenet"

batch_size = 128
epochs = 400
#epochs = 7
test_loss_min = 1e5
epoch_loss_min = []
best_accuracy = 0.7
keep_prob = 0.5


path = os.getcwd()+'/'+SAVE_DIRECTORY_NAME
if not os.path.isdir(path):
    os.mkdir(path, 0o0777)

# setup the initialisation operator
init_op = tf.global_variables_initializer()

# add ops to save and restore all the variables
saver = tf.train.Saver(max_to_keep=2)

def print_settings():
    print("########################")
    if not LOAD:
        print ("LOAD=False.\t No weights loaded")
    else:
        print("LOAD=True.\t Model weights loaded from: " + LOAD_DIRECTORY_NAME)
    if not SAVE:
        print ("SAVE=False.\t Model will not be saved")
    else:
        print("SAVE=True.\tModel weights saved at: " + SAVE_DIRECTORY_NAME)
    print ("GPU number: ", GPU)
    print ("coherence is: ", graph.coherence_coef)
    print ("weight decay is: ", graph.weight_decay)
    print ("dropout keep_prob: ", keep_prob)
    print("########################")

print_settings()

def randomize(dataset, cls):
  permutation = np.random.permutation(cls.shape[0])
  dataset = np.asarray(dataset)
  shuffled_dataset = dataset[permutation,:]
  shuffled_cls = cls[permutation]
  return shuffled_dataset, shuffled_cls

def plot_dist(sess):
    mat = sess.run(graph.layer.kernel)
    co_mat = np.matmul(mat.T, mat)
    values = np.array([co_mat[i,j] for i in range(co_mat.shape[0]) for j in range(co_mat.shape[1]) if i!=j])
    plt.hist(values, bins=20)
    title = "dropout keep_prob: " + str(keep_prob).replace(".","_")+ " , coherence coef: " +str(graph.coherence_coef).replace(".","_")
    plt.title(title+" var: "+str(np.var(values))+ "absolute var: "+str(np.var(np.absolute(values))))
    plt.savefig("histogram - "+title)
    np.save(title,np.array(values))

with tf.Session() as sess:
    if LOAD:
        saver.restore(sess,tf.train.latest_checkpoint(os.path.join(os.getcwd(),LOAD_DIRECTORY_NAME), "checkpoint"))
        graph.lr_init *= 0.1
        print("Weights loaded")
    else:
        sess.run(init_op)

    X_train, y_train = mnist_reader.load_mnist("data", kind='train')
    X_test, y_test = mnist_reader.load_mnist("data", kind='t10k')
    print("sample shape: ", X_train[0].shape)

    #reshape & pad
    X_train = np.reshape(X_train,(-1,28,28,1))
    X_test = np.reshape(X_test,(-1,28,28,1))
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    print("sample shape: ", X_train[0].shape)

    total_batch_train = int(math.ceil(len(X_train) / batch_size))
    total_batch_test = int(math.ceil(len(X_test) / batch_size))
    train_writer = tf.summary.FileWriter(logs_path + '/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(logs_path + '/test', graph=tf.get_default_graph())

    for epoch in range(epochs):
        learning_rate = graph.set_lr(epoch)
        #training
        if KEEP_TRAINING:
            avg_loss_train = 0
            avg_acc_train = 0
            avg_l2_loss = 0
            avg_coherence_loss = 0
            images_train, cls_train = shuffle(X_train,y_train)
            for i in range(total_batch_train):
                offset = i * batch_size
                batch_data = images_train[offset:(offset + batch_size), :]
                batch_labels = cls_train[offset:(offset + batch_size)]
                feed_dict_train = {graph.x: batch_data, graph.y: batch_labels, graph.learning_rate: learning_rate, graph.dropout: keep_prob}
                train_acc, _, loss_train, l2, co_loss = sess.run([graph.accuracy, graph.optimizer, graph.cost, graph.l2_loss, graph.coherence_loss], feed_dict=feed_dict_train)
                avg_loss_train += loss_train / total_batch_train
                avg_acc_train += train_acc / total_batch_train
                avg_l2_loss += l2 / total_batch_train
                avg_coherence_loss += co_loss / total_batch_train

            print("Epoch", (epoch + 1),":", "train loss =", "{:.3f}".format(avg_loss_train), "train accuracy =", "{:.2f}".format(avg_acc_train*100),\
                  "l2 loss = ", "{:.3f}".format(avg_l2_loss), " coherence loss = ", "{:.3f}".format(avg_coherence_loss))

        avg_loss_test = 0
        avg_acc_test = 0
        for i in range(total_batch_test):
            offset = i * batch_size
            batch_data_test = X_test[offset:(offset + batch_size), :]
            batch_labels_test = y_test[offset:(offset + batch_size)]
            feed_dict_test = {graph.x: batch_data_test, graph.y: batch_labels_test, graph.learning_rate: learning_rate, graph.dropout: 1.}
            test_acc, loss_test , co = sess.run([graph.accuracy, graph.cost, graph.coherence], feed_dict=feed_dict_test) #for test loss is considered without the weight decay
            avg_loss_test += loss_test / total_batch_test
            avg_acc_test += test_acc / total_batch_test
        print("Epoch", (epoch + 1),":", "test loss: {:.3f}".format(avg_loss_test), "test accuracy: {:.2f}".format(avg_acc_test*100), "coherence: {:.4f}".format(co))
        if best_accuracy < avg_acc_test:
            if SAVE:
                saver.save(sess, os.path.join(path, 'my_model'), global_step=epoch, write_meta_graph=False)
                print("model saved")
            else:
                print("model not saved")
            best_accuracy = avg_acc_test
            best_epoch=epoch

    plot_dist(sess)
    print("\nTraining complete!")
    print("Best acc is ", best_accuracy*100," at epoch ", best_epoch)
    print("Dropout = ", keep_prob, ", Weight decay = ", graph.weight_decay, ", coherence =", graph.coherence_coef)