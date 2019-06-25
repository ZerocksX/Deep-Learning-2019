import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from faks.du.lab1 import data


class TFLogreg:
    def __init__(self, D, C, param_delta=0.5, param_lambda=1e-2):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
           - param_delta: training step
        """

        # definicija podataka i parametara:
        # definirati self.X, self.Yoh_, self.W, self.b
        # ...

        self.X = tf.placeholder(tf.float32, shape=(None, D))
        self.Yoh_ = tf.placeholder(tf.float32, shape=(None, C))
        self.W = tf.Variable(tf.random_normal((D, C)))
        self.b = tf.Variable(tf.zeros((C,)))
        self.param_delta = param_delta
        self.param_lambda = param_lambda

        # formulacija modela: izračunati self.probs
        #   koristiti: tf.matmul, tf.nn.softmax
        # ...

        self.score = tf.matmul(self.X, self.W) + self.b
        self.probs = tf.nn.softmax(self.score)

        # formulacija gubitka: self.loss
        #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
        # ...
        # self.loss = -tf.reduce_sum(tf.log(self.probs) * self.Yoh_)
        self.loss = -tf.reduce_sum(tf.log(self.probs) * self.Yoh_) + tf.norm(self.W, ord=2) * self.param_lambda

        # formulacija operacije učenja: self.train_step
        #   koristiti: tf.train.GradientDescentOptimizer,
        #              tf.train.GradientDescentOptimizer.minimize
        # ...
        self.optimizer = tf.train.GradientDescentOptimizer(self.param_delta)
        grads_and_vars = self.optimizer.compute_gradients(self.loss, [self.W, self.b])
        batch_size = tf.cast(tf.shape(X)[0], dtype=tf.float32)
        capped_grads_and_vars = [(gv[0] / batch_size, gv[1]) for gv in grads_and_vars]
        self.train_step = self.optimizer.apply_gradients(capped_grads_and_vars)

        self.print = tf.Print(self.probs, [self.probs, self.loss], 'Log: ')

        # instanciranje izvedbenog konteksta: self.session
        #   koristiti: tf.Session
        # ...

        self.session = tf.Session()
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
            loss, _, W, b, _ = self.session.run([self.loss, self.train_step, self.W, self.b, self.print],
                                                feed_dict={self.X: X, self.Yoh_: Yoh_})
            print(i, loss)
        return W, b

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        #   koristiti: tf.Session.run
        return self.session.run([self.probs], feed_dict={self.X: X})[0]


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    tf.set_random_seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gauss_2d(3, 100)
    Yoh_ = data.class_to_onehot(Y_)

    # izgradi graf:
    tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.5, 1e-2)

    # nauči parametre:
    tflr.train(X, Yoh_, 1000)

    # dohvati vjerojatnosti na skupu za učenje
    probs = tflr.eval(X)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print('acc',accuracy)
    print('recall', recall)
    print('precision', precision)

    # iscrtaj rezultate, decizijsku plohu

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: np.argmax(tflr.eval(x), axis=1), rect, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    data.plt.show()
