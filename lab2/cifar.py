import os
import math
import pickle
import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.io
from lab1 import data
import matplotlib.pyplot as plt


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


DATA_DIR = 'datasets/CIFAR10/cifar-10-batches-py/'
SAVE_DIR = 'source/fer/out/cifar'

config = {
    'max_epochs': 32,
    'batch_size': 100,
    'save_dir': SAVE_DIR,
    'lr_policy': {1: {'lr': 5e-2}, 8: {'lr': 5e-3}, 12: {'lr': 1e-3}, 24: {'lr': 1e-4}},
    'weight_decay': 1e-3,
    'conv1sz': 16,
    'conv2sz': 32,
    'conv3sz': 56,
    'fc3sz': 256,
    'fc4sz': 128,
    'fc5sz': 10,
}
inputs = tf.placeholder(tf.float32, (None, 32, 32, 3), 'X')
labels = tf.placeholder(tf.float32, (None, 10), 'Yoh')
lr = tf.placeholder(tf.float32, name='LR')


def draw_image(img, mean, std):
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[3]
    num_channels = w.shape[2]
    k = w.shape[0]
    assert w.shape[0] == w.shape[1]
    w = w.reshape(k, k, num_channels, num_filters)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def get_model(inputs, labels, lr, config):
    net = []
    conv1sz = config['conv1sz']
    conv2sz = config['conv2sz']
    conv3sz = config['conv3sz']
    fc3sz = config['fc3sz']
    fc4sz = config['fc4sz']
    fc5sz = config['fc5sz']
    weight_decay = config['weight_decay']
    net += [tf.layers.conv2d(inputs, conv1sz, 5, strides=1, padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1')]
    net += [tf.layers.max_pooling2d(net[-1], 3, 2, 'SAME', name='pool1')]
    net += [tf.layers.conv2d(inputs, conv2sz, 7, strides=1, padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2')]
    net += [tf.layers.max_pooling2d(net[-1], 3, 2, 'SAME', name='pool2')]
    net += [tf.layers.conv2d(inputs, conv3sz, 5, strides=1, padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv3')]
    net += [tf.layers.max_pooling2d(net[-1], 3, 2, 'SAME', name='pool3')]
    net += [tf.layers.flatten(net[-1], 'flat1')]
    net += [tf.layers.dense(net[-1], fc3sz, activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='dense1')]
    net += [tf.layers.dense(net[-1], fc4sz, activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='dense2')]
    net += [tf.layers.dense(net[-1], fc5sz, activation=None,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name='dense3')]
    loss = tf.losses.softmax_cross_entropy(labels, net[-1], scope='loss1') + tf.losses.get_regularization_loss()
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, tf.train.get_global_step())
    return net, loss, train_op


def evaluate(session, name, x, y, net, loss, config):
    print("\nRunning evaluation: ", name)
    batch_size = config['batch_size']
    num_examples = x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    m = None
    loss_sum = 0
    for i in range(num_batches):
        batch_x = x[i * batch_size:(i + 1) * batch_size, :]
        batch_y = y[i * batch_size:(i + 1) * batch_size, :]

        feed_dict = {
            inputs: batch_x,
            labels: batch_y
        }
        logits, loss_val = session.run([net[-1], loss], feed_dict=feed_dict)
        loss_sum += loss_val
        Y = np.argmax(logits, axis=1)
        Y_ = np.argmax(batch_y, axis=1)
        acc, recpre, M = data.eval_perf_multi(Y, Y_, 10)
        if m is None:
            m = np.array(M).reshape((10, 10))
        else:
            m += np.array(M).reshape((10, 10))
    print('Confusion')
    print(m)
    rp = []
    for i in range(10):
        tp_i = m[i, i]
        fn_i = np.sum(m[i, :]) - tp_i
        fp_i = np.sum(m[:, i]) - tp_i
        tn_i = np.sum(m) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        rp.append((recall_i, precision_i))

    print('(Recall, Precision)', rp)

    accuracy = np.trace(m) / np.sum(m)
    print('Accuracy: ', accuracy)
    return loss_sum / num_batches, accuracy


def train(session, train_x, train_y, valid_x, valid_y, net, loss, train_op, config):
    session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    lr_policy = config['lr_policy']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']
    num_examples = train_x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []
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
            feed_dict = {
                inputs: batch_x,
                labels: batch_y
            }
            logits, loss_val = session.run([net[-1], loss], feed_dict=feed_dict)
            # loss_val = loss.forward(logits, batch_y)
            # compute classification accuracy
            yp = np.argmax(logits, 1)
            yt = np.argmax(batch_y, 1)
            cnt_correct += (yp == yt).sum()
            # grads = backward_pass(logits_, loss, logits, batch_y)
            # sgd_update_params(grads, solver_config)

            topval, _ = sess.run([train_op, loss], feed_dict={
                inputs: batch_x,
                labels: batch_y,
                lr: solver_config['lr']
            })

            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i * batch_size, num_examples, loss_val))
            if i % 100 == 0:
                weights = sess.run([tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel:0')[0]])[0]
                draw_conv_filters(epoch, i * batch_size, weights, save_dir)
                # draw_conv_filters(epoch, i*batch_size, net[3])
            if i > 0 and i % 50 == 0:
                print("Train accuracy = %.2f" % (cnt_correct / ((i + 1) * batch_size) * 100))
        print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
        train_loss, train_acc = evaluate(session, "Train", train_x, train_y, net, loss, config)
        valid_loss, valid_acc = evaluate(session, "Validation", valid_x, valid_y, net, loss, config)
        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [valid_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [valid_acc]
        plot_data['lr'] += [solver_config['lr']]
        plot_training_progress(SAVE_DIR, plot_data)


img_height = 32
img_width = 32
num_channels = 3

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# sess = tf.Session(config=tf_config)
sess = tf.Session()

# sess.run(tf.initialize_all_variables())

net, loss, train_op = get_model(inputs, labels, lr, config)
train(sess, train_x, data.class_to_onehot(train_y), valid_x, data.class_to_onehot(valid_y), net, loss, train_op, config)
evaluate(sess, 'TEST', test_x, data.class_to_onehot(test_y), net, loss, config)
imgs = {}
losses = []
# for i in range(len(test_y)):
#     x, y_ = test_x[i], test_y[i]
#     y, l = sess.run([net[-1], loss], feed_dict={inputs: np.array([x]), labels: data.class_to_onehot(np.array([y_]), 10)})
#     imgs[i] = x
#     losses.append(l)
h, l = sess.run([net[-1], loss], feed_dict={inputs: test_x, labels: data.class_to_onehot(test_y, 10)})
y = np.exp(h) / np.sum(np.exp(h), axis=1).reshape(len(test_x), 1)
losses = np.sum(np.log(y) * data.class_to_onehot(test_y, 10), axis=1)
failed_ims = sorted(enumerate(test_x), key=lambda kv: losses[kv[0]])
# print(failed_ims)
for i in range(20):
    draw_image(failed_ims[i][1].reshape((32, 32, 3)), data_mean, data_std)
