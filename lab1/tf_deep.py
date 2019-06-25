import operator
import os
import timeit
from functools import reduce
from time import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from faks.du.lab1 import data


class TFDeep:
    def __init__(self, layers, param_delta=0.5, param_lambda=1e-2):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
           - param_delta: training step
        """

        # definicija podataka i parametara:
        # definirati self.X, self.Yoh_, self.W, self.b
        # ...
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

        # formulacija operacije učenja: self.train_step
        #   koristiti: tf.train.GradientDescentOptimizer,
        #              tf.train.GradientDescentOptimizer.minimize
        # ...
        self.optimizer = tf.train.GradientDescentOptimizer(self.param_delta)
        grads_and_vars = self.optimizer.compute_gradients(self.loss, [*self.W, *self.b])
        batch_size = tf.cast(tf.shape(self.X)[0], dtype=tf.float32)
        capped_grads_and_vars = [(gv[0] / batch_size, gv[1]) for gv in grads_and_vars]
        self.train_step = self.optimizer.apply_gradients(capped_grads_and_vars)
        # self.train_step = self.optimizer.minimize(self.loss)

        # self.print = tf.Print(self.probs, [self.probs, self.loss, self.h], 'Log: ')

        # instanciranje izvedbenog konteksta: self.session
        #   koristiti: tf.Session
        # ...

        with tf.name_scope('scalars'):
            tf_loss_summary = tf.summary.scalar('loss', self.loss)
        self.saver = tf.train.Saver()
        self.id = 'tf_deep_' + str(int(time()))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True

        self.session = tf.Session(config=config)

        self.summ_writer = tf.summary.FileWriter(os.path.join('/tmp/tensorflow/summaries', self.id), self.session.graph)
        self.summaries = tf.summary.merge([tf_loss_summary])
        self.session.run(tf.initialize_all_variables())

    def train(self, X, Yoh_, param_niter):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        # incijalizacija parametara
        #   koristiti: tf.initializers.global_variables
        # ...
        tf.initializers.global_variables()

        # optimizacijska petlja
        #   koristiti: tf.Session.run
        # ...
        W, b = None, None
        for i in range(param_niter):
            loss, _, W, b, sum = self.session.run([self.loss, self.train_step, self.W, self.b, self.summaries],
                                             feed_dict={self.X: X, self.Yoh_: Yoh_})
            self.summ_writer.add_summary(sum, i)
            if (i % 100) == 0:
                print(i, loss)
                self.saver.save(self.session, '/tmp/data/' + self.id + '/my_model_%d' % i)
        self.saver.save(self.session, '/tmp/data/' + self.id + '/my_model')
        return W, b

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        #   koristiti: tf.Session.run
        return self.session.run([self.probs], feed_dict={self.X: X})[0]

    def count_params(self):
        print(tf.trainable_variables(),
              sum(map(lambda x: reduce(operator.mul, map(lambda y: int(y), x.shape)), tf.trainable_variables())))

def main():
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    tf.set_random_seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    Yoh_ = data.class_to_onehot(Y_)

    # izgradi graf:
    tflr = TFDeep([2, 3, 2, 2], 0.1, 1e-4)
    tflr.count_params()

    # nauči parametre:
    tflr.train(X, Yoh_, int(1e4))

    # dohvati vjerojatnosti na skupu za učenje
    probs = tflr.eval(X)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print('acc', accuracy)
    print('recall', recall)
    print('precision', precision)

    # iscrtaj rezultate, decizijsku plohu

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: np.argmax(tflr.eval(x), axis=1), rect, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    data.plt.show()

def cpu():
    with tf.device('/cpu:0'):
        main()

def gpu():
    # with tf.device('/gpu:0'):
    main()

if __name__ == "__main__":
    # cpu_time = timeit.timeit('cpu()', number=1, setup="from __main__ import cpu")
    # gpu_time = timeit.timeit('gpu()', number=1, setup="from __main__ import gpu")
    # print(cpu_time)
    # print(gpu_time)
    # print(cpu_time / gpu_time)
    main()