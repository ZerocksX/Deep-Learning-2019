import operator
import os
import sklearn
import timeit
from functools import reduce
from time import time

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from faks.du.lab1 import tf_deep
from faks.du.lab1 import data
from faks.du.lab1 import ksvm_wrap
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_string('data_dir',
                           '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(
    tf.app.flags.FLAGS.data_dir, one_hot=True)
N = mnist.train.images.shape[0]
D = mnist.train.images.shape[1]
C = mnist.train.labels.shape[1]
print(N, D, C)

digits = [mnist.test.images[np.nonzero(mnist.test.labels.T[i] == 1)[0][0]] for i in range(10)]


# for digit in digits:
#     plt.imshow(digit.reshape((28, 28)), cmap=plt.get_cmap('gray'), vmin=0, vmax=1.)
#     plt.show()

class TFDeep:
    def __init__(self, layers, param_delta=0.5, param_lambda=1e-2, decay_rate=1 - 1e-4, decay_stepsi=1):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
           - param_delta: training step
        """

        # definicija podataka i parametara:
        # definirati self.X, self.Yoh_, self.W, self.b
        # ...
        self.best_W = np.array([])
        self.best_b = np.array([])
        self.best_loss = 1e9
        D = layers[0]
        C = layers[-1]
        self.N = len(layers) - 1
        self.X = tf.placeholder(tf.float32, shape=(None, D))
        self.Yoh_ = tf.placeholder(tf.float32, shape=(None, C))
        self.W = []
        self.b = []
        for i in range(self.N):
            self.W.append(tf.Variable(tf.random_normal((layers[i], layers[i + 1])), name='W' + str(i)))
            self.b.append(tf.Variable(tf.zeros((layers[i + 1],)), name='b' + str(i)))
        self.param_delta = param_delta
        self.param_lambda = param_lambda
        self.decay_stepsi = decay_stepsi
        self.decay_rate = decay_rate
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)

        self.learning_rate = tf.train.exponential_decay(self.param_delta, self.global_step,
                                                        decay_steps=self.decay_stepsi,
                                                        decay_rate=self.decay_rate)
        batch_size = tf.cast(tf.shape(self.X)[0], dtype=tf.float32)

        # formulacija modela: izračunati self.probs
        #   koristiti: tf.matmul, tf.nn.softmax
        # ...
        self.h = []
        self.f = [tf.nn.sigmoid if i != self.N - 1 else tf.nn.softmax for i in range(self.N)]
        input = tf.cast(self.X, dtype=tf.float32)
        for i in range(0, self.N - 1):
            self.h.append(tf.cast(self.f[i](tf.matmul(input, self.W[i]) + self.b[i]), dtype=tf.float32))
            input = self.h[i]

        self.score = tf.matmul(input, self.W[-1]) + self.b[-1]
        self.probs = self.f[-1](self.score)

        # formulacija gubitka: self.loss
        #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
        # ...
        self.loss = -tf.reduce_sum(tf.log(self.probs) * self.Yoh_)
        for i in range(self.N):
            self.loss += tf.norm(self.W[i], ord=2) * param_lambda

        self.average_loss = self.loss / batch_size

        # formulacija operacije učenja: self.train_step
        #   koristiti: tf.train.GradientDescentOptimizer,
        #              tf.train.GradientDescentOptimizer.minimize
        # ...
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # grads_and_vars = self.optimizer.compute_gradients(self.loss, [*self.W, *self.b])
        # capped_grads_and_vars = [(gv[0] / batch_size, gv[1]) for gv in grads_and_vars]
        # self.train_step = self.optimizer.apply_gradients(capped_grads_and_vars)
        self.train_step = self.optimizer.minimize(self.average_loss)

        # self.print = tf.Print(self.probs, [self.probs, self.loss, self.h], 'Log: ')

        # instanciranje izvedbenog konteksta: self.session
        #   koristiti: tf.Session
        # ...

        with tf.name_scope('scalars'):
            self.tf_loss_summary = tf.summary.scalar('loss', self.loss)
            self.tf_avg_loss_summary = tf.summary.scalar('average-loss', self.average_loss)
            self.tf_learning_rate_summary = tf.summary.scalar('learning-rate', self.learning_rate)
        self.saver = tf.train.Saver()
        self.id = 'tf_deep_' + str(int(time()))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True

        self.session = tf.Session(config=config)

        self.summ_writer = tf.summary.FileWriter(os.path.join('/tmp/tensorflow/summaries', self.id), self.session.graph)
        self.summaries = tf.summary.merge([self.tf_loss_summary, self.tf_learning_rate_summary])
        self.session.run(tf.initialize_all_variables())

    def train(self, X, Yoh_, param_niter):
        tf.initializers.global_variables()
        for i in range(param_niter):
            self._train(X, Yoh_)
            loss, sum, W, b = self.session.run(
                [self.average_loss, self.tf_avg_loss_summary],
                feed_dict={self.X: X, self.Yoh_: Yoh_})
            self.summ_writer.add_summary(sum, i)
            print("Epoch %d finished with average loss %f" % (i, loss))

        self.saver.save(self.session, '/tmp/data/' + self.id + '/my_model')

    def _train(self, X, Yoh_):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        # incijalizacija parametara
        #   koristiti: tf.initializers.global_variables
        # ...

        # optimizacijska petlja
        #   koristiti: tf.Session.run
        # ...

        loss, _, sum, step = self.session.run(
            [self.loss, self.train_step, self.summaries, self.increment_global_step_op],
            feed_dict={self.X: X, self.Yoh_: Yoh_})
        self.summ_writer.add_summary(sum, step)
        if (step % 500) == 0:
            print(step, loss)
            self.saver.save(self.session, '/tmp/data/' + self.id + '/my_model_%d' % step)

    def train_mb(self, X, Yoh_, param_niter, batch_size=2 ** 7):
        tf.initializers.global_variables()
        X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X, Yoh_, test_size=1 / 5, random_state=100)
        batch_size = min(X_train.shape[0], batch_size)
        for epoch in range(param_niter):
            for i in range(X_train.shape[0] // batch_size):
                batch_index = np.random.randint(0, X_train.shape[0], batch_size)
                X_batch = X_train[batch_index, :]
                Yoh__batch = Y_train[batch_index, :]
                self._train(X_batch, Yoh__batch)
            loss, sum, W, b = self.session.run(
                [self.average_loss, self.tf_avg_loss_summary, self.W, self.b],
                feed_dict={self.X: X_val, self.Yoh_: Y_val})
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_W = W[:]
                self.best_b = b[:]
            self.summ_writer.add_summary(sum, epoch)
            print("Epoch %d finished with average loss %f" % (epoch, loss))

        self.saver.save(self.session, '/tmp/data/' + self.id + '/my_model')

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        #   koristiti: tf.Session.run
        return self.session.run([self.probs], feed_dict={self.X: X})[0]

    def eval_with_weights(self, X, W, b):
        N = len(W)
        h = []
        f = self.f
        input = X
        for i in range(0, N - 1):
            h.append(tf.Session().run([f[i](np.add(np.matmul(input, W[i]), b[i].T))])[0])
            input = h[i]

        probs = tf.Session().run([f[-1](np.add(np.matmul(input, W[-1]), b[-1].T))])[0]
        return probs

    def count_params(self):
        print(tf.trainable_variables(),
              sum(map(lambda x: reduce(operator.mul, map(lambda y: int(y), x.shape)), tf.trainable_variables())))


def main():
    layers_conf = [
        [784, 10],
        [784, 100, 10],
        [784, 100, 100, 10],
        [784, 100, 100, 100, 10]
    ]
    results = []
    for layers in layers_conf:
        print(layers)
        clf = TFDeep(layers, 5e-3, 1e-4)
        clf.train_mb(mnist.train.images, mnist.train.labels, 25)
        for digit in digits:
            probs = clf.eval(np.array([[*digit]]))
            print(np.argmax(probs, axis=1))
        Y_ = np.argmax(clf.eval(mnist.test.images), axis=1)
        Y = np.argmax(mnist.test.labels, axis=1)
        # acc, pr, m = data.eval_perf_multi(Y, Y_)
        # print(acc)
        # print(sum([p for p, r in pr]) / 10, [p for p, r in pr])
        # print(sum([r for p, r in pr]) / 10, [r for p, r in pr])
        # print(m)
        # results.append(
        #     (
        #         acc,
        #         (sum([p for p, r in pr]) / 10, [p for p, r in pr]),
        #         (sum([r for p, r in pr]) / 10, [r for p, r in pr]),
        #         m
        #     )
        # )

        Y_best = np.argmax(clf.eval_with_weights(mnist.test.images, clf.best_W, clf.best_b), axis=1)
        acc, pr, m = data.eval_perf_multi(Y, Y_best)
        print(acc)
        print(sum([p for p, r in pr]) / 10, [p for p, r in pr])
        print(sum([r for p, r in pr]) / 10, [r for p, r in pr])
        print(m)
        results.append(
            (
                acc,
                (sum([p for p, r in pr]) / 10, [p for p, r in pr]),
                (sum([r for p, r in pr]) / 10, [r for p, r in pr]),
                m
            )
        )

        if clf.N == 1:
            W = clf.best_W
            for weights in W[0].T:
                plt.imshow(weights.reshape((28, 28)), cmap=plt.get_cmap('gray'), vmin=min(weights), vmax=max(weights))
                plt.show()
    for l, r in zip(layers_conf, results):
        print(l)
        for i in r:
            print(i)


def cpu():
    with tf.device('/cpu:0'):
        main()


def gpu():
    # with tf.device('/gpu:0'):
    main()


def svm_sol():
    for kernel in ['linear', 'rbf']:
        print(kernel)
        clf = ksvm_wrap.KSVMWrap(mnist.train.images, np.argmax(mnist.train.labels, axis=1), kernel=kernel)
        Y_ = clf.predict(mnist.test.images)
        Y = np.argmax(mnist.test.labels, axis=1)
        print(Y, Y_)
        acc, pr, m = data.eval_perf_multi(Y, Y_)
        print(acc)
        print(sum([p for p, r in pr]) / 10, [p for p, r in pr])
        print(sum([r for p, r in pr]) / 10, [r for p, r in pr])
        print(m)


if __name__ == "__main__":
    # cpu_time = timeit.timeit('cpu()', number=1, setup="from __main__ import cpu")
    # print(cpu_time)
    # gpu_time = timeit.timeit('gpu()', number=1, setup="from __main__ import gpu")
    # print(gpu_time)
    # print(cpu_time / gpu_time)
    main()
    # svm_sol()
