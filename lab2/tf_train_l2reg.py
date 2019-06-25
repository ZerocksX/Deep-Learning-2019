import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import os
import time
import skimage as ski
import skimage.io
from lab1 import data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
# sess = tf.Session()

DATA_DIR = 'datasets/MNIST/'
SAVE_DIR = "source/fer/out/tf/"

config = {
    'max_epochs': 16,
    'batch_size': 500,
    'save_dir': SAVE_DIR,
    'lr_policy': {1: {'lr': 1e-2}, 2: {'lr': 5e-3}, 4: {'lr': 1e-3}, 8: {'lr': 1e-4}},
    'weight_decay': 1e-1,
    'conv1sz': 16,
    'conv2sz': 32,
    'fc3sz': 512,
}

net = []

lr = tf.placeholder(dtype=tf.float32)


def build_model(inputs, labels, num_classes):
    global net
    weight_decay = config['weight_decay']
    conv1sz = config['conv1sz']
    conv2sz = config['conv2sz']
    fc3sz = config['fc3sz']
    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net += [layers.convolution2d(inputs, conv1sz, scope='conv1')]
        net += [layers.max_pool2d(net[-1], kernel_size=5, stride=1, padding='SAME', scope='pool1')]
        net += [layers.convolution2d(net[-1], conv2sz, scope='conv2')]
        net += [layers.max_pool2d(net[-1], kernel_size=5, stride=1, padding='SAME', scope='pool2')]

    with tf.contrib.framework.arg_scope([layers.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        # sada definiramo potpuno povezane slojeve
        # ali najprije prebacimo 4D tenzor u matricu
        net += [layers.flatten(net[-1])]
        net += [layers.fully_connected(net[-1], fc3sz, scope='fc3')]

    logits = layers.fully_connected(net[-1], num_classes, activation_fn=None, scope='logits')
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, scope='loss') + tf.losses.get_regularization_loss()
    # batch_size = tf.cast(tf.shape(inputs)[0], dtype=tf.float32)
    # loss /= batch_size

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )

    return logits, loss, train_op


def draw_conv_filters(epoch, step, layer, save_dir):
    C = 10
    W = sess.run([tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/weights:0')[0]])[0] # (5,5,1,16)
    w = []
    for i in range(16):
        w.append(np.array([W[:,:,:,i].reshape((5,5))]))
    w = np.array(w)
    # w = [:,:,:,1].reshape((5,5))
    num_filters = w.shape[0]
    k = w.shape[2]
    w = w.reshape((num_filters, 1, k, k))
    # num_filters = 16
    # k = 5
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    # for i in range(C):
    for i in range(1):
        img = np.zeros([height, width])
        for j in range(num_filters):
            r = int(j / cols) * (k + border)
            c = int(j % cols) * (k + border)
            img[r:r + k, c:c + k] = w[j, i]
        filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % ('conv1', epoch, step, i)
        ski.io.imsave(os.path.join(save_dir, filename), img)


def forward_pass(net, loss, batch_x, batch_y):
    global inputs, labels
    return sess.run([net, loss], feed_dict={
        inputs: batch_x,
        labels: batch_y
    })


def evaluate(name, x, y, net, loss, config):
    print("\nRunning evaluation: ", name)
    batch_size = config['batch_size']
    num_examples = x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    cnt_correct = 0
    loss_avg = 0
    for i in range(num_batches):
        batch_x = x[i * batch_size:(i + 1) * batch_size, :]
        batch_y = y[i * batch_size:(i + 1) * batch_size, :]
        logits, loss_val = forward_pass(net, loss, batch_x, batch_y)
        yp = np.argmax(logits, 1)
        yt = np.argmax(batch_y, 1)
        cnt_correct += (yp == yt).sum()
        loss_avg += loss_val
        # print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
    valid_acc = cnt_correct / num_examples * 100
    loss_avg /= num_batches
    print(name + " accuracy = %.2f" % valid_acc)
    print(name + " avg loss = %.2f\n" % loss_avg)


def train(train_x, train_y, valid_x, valid_y, logits_, loss, config):
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    lr_policy = config['lr_policy']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']
    num_examples = train_x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    for epoch in range(1, max_epochs + 1):
        if epoch in lr_policy:
            solver_config = lr_policy[epoch]
        cnt_correct = 0
        # for i in range(num_batches):
        # shuffle the data at the beggining of each epoch
        permutation_idx = np.random.permutation(num_examples)
        train_x = train_x[permutation_idx]
        train_y = train_y[permutation_idx]
        # for i in range(100):
        for i in range(num_batches):
            # store mini-batch to ndarray
            batch_x = train_x[i * batch_size:(i + 1) * batch_size, :]
            batch_y = train_y[i * batch_size:(i + 1) * batch_size, :]
            logits, loss_val = forward_pass(logits_, loss, batch_x, batch_y)
            # loss_val = loss.forward(logits, batch_y)
            # compute classification accuracy
            yp = np.argmax(logits, 1)
            yt = np.argmax(batch_y, 1)
            cnt_correct += (yp == yt).sum()
            # grads = backward_pass(logits_, loss, logits, batch_y)
            # sgd_update_params(grads, solver_config)

            topval, loss_val, layer = sess.run([train_op, loss, net[0]], feed_dict={
                inputs: batch_x,
                labels: batch_y,
                lr: solver_config['lr']
            })

            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i * batch_size, num_examples, loss_val))
            if i % 100 == 0:
                draw_conv_filters(epoch, i * batch_size,layer, save_dir)
                # draw_conv_filters(epoch, i*batch_size, net[3])
            if i > 0 and i % 50 == 0:
                print("Train accuracy = %.2f" % (cnt_correct / ((i + 1) * batch_size) * 100))
        print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
        evaluate("Validation", valid_x, valid_y, logits_, loss, config)
    return logits_


if __name__ == '__main__':
    np.random.seed(int(time.time() * 1e6) % 2 ** 31)
    dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
    train_x = dataset.train.images
    train_x = train_x.reshape([-1, 28, 28, 1])
    train_y = dataset.train.labels
    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 28, 28, 1])
    valid_y = dataset.validation.labels
    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 28, 28, 1])
    test_y = dataset.test.labels
    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean

    inputs = tf.placeholder(tf.float32, (None, 28, 28, 1), 'inputs')
    n_classes = 10
    labels = tf.placeholder(tf.float32, (None, n_classes), 'labels')
    net, loss, train_op = build_model(inputs, labels, n_classes)
    train(train_x, train_y, valid_x, valid_y, net, loss, config)
    evaluate('Test', test_x, test_y, net, loss, config)
    sess.close()
