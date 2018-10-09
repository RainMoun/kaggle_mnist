#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import pandas as pd

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

LayersConfig.tf_dtype = tf.float16  # tf.float32  tf.float16

train_data_file = './data/train.csv'
test_data_file = './data/test.csv'

train_data = pd.read_csv(train_data_file).as_matrix().astype(np.uint8)
test_data = pd.read_csv(test_data_file).as_matrix().astype(np.uint8)


def extract_images_and_labels(dataset, validation=False):
    # 需要将数据转化为[image_num, x, y, depth]格式
    images = dataset[:, 1:].reshape(-1, 28, 28, 1)

    labels_dense = dataset[:, 0]
    if validation:
        num_images = images.shape[0]
        divider = num_images - 200
        return images[:divider], labels_dense[:divider], images[divider+1:], labels_dense[divider+1:]
    else:
        return images, labels_dense


def extract_images(dataset):
    return dataset.reshape(-1, 28, 28, 1)


train_images, train_labels, val_images, val_labels = extract_images_and_labels(train_data, validation=True)
test_images = extract_images(test_data)


sess = tf.InteractiveSession()

batch_size = 128

x = tf.placeholder(LayersConfig.tf_dtype, shape=[None, 28, 28, 1])
y_ = tf.placeholder(tf.int64, shape=[None])


def model(x, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        n = InputLayer(x, name='input')
        # cnn
        n = Conv2d(n, 32, (5, 5), (1, 1), padding='SAME', name='cnn1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, name='bn1')
        n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='pool1')
        n = Conv2d(n, 64, (5, 5), (1, 1), padding='SAME', name='cnn2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, name='bn2')
        n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='pool2')
        # mlp
        n = FlattenLayer(n, name='flatten')
        n = DropoutLayer(n, 0.5, True, is_train, name='drop1')
        n = DenseLayer(n, 256, act=tf.nn.relu, name='relu1')
        n = DropoutLayer(n, 0.5, True, is_train, name='drop2')
        n = DenseLayer(n, 10, act=None, name='output')
    return n


# define inferences
net_train = model(x, is_train=True, reuse=False)
net_test = model(x, is_train=False, reuse=True)

net_train.print_params(False)

# cost for training
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')

# cost and accuracy for evalution
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, LayersConfig.tf_dtype))

# define the optimizer
train_params = tl.layers.get_variables_with_name('model', train_only=True, printable=False)
# for float16 epsilon=1e-4 see https://stackoverflow.com/questions/42064941/tensorflow-float16-support-is-broken
# for float32 epsilon=1e-08
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-4,
                                  use_locking=False).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# train the network
n_epoch = 500
print_freq = 1

for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(train_images, train_labels, batch_size, shuffle=True):
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(train_images, train_labels, batch_size, shuffle=True):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))
        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(val_images, val_labels, batch_size, shuffle=True):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("   val loss: %f" % (val_loss / n_batch))
        print("   val acc: %f" % (val_acc / n_batch))

f = open('./result/prediction.csv', 'w+')
f.write('ImageId,Label\n')


def get_batchs(data, batch_size):
    size = data.shape[0]
    for i in range(size//batch_size):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:]
        else:
            yield data[i*batch_size:(i+1)*batch_size]


batchs = get_batchs(test_images, 50)
i = 1
for test_image in batchs:
    prediction = tf.argmax(y2, 1)
    test_labels = prediction.eval(feed_dict={x: test_image})
    for label in test_labels:
        f.write(str(i) + ',' +str(label) + '\n')
        i += 1
f.close()
