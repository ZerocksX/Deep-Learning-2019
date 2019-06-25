from faks.du.lab1 import data
import numpy as np

param_niter = int(1e5)
delta_big = 0.01
delta_small = 0.0001
get_param_delta = lambda iteration: (iteration / param_niter) * delta_small + (
        (param_niter - iteration) / param_niter) * delta_big
# get_param_delta = lambda x : delta_big

param_delta = 0.05
param_lambda = 1e-3


# stabilni softmax
def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs


def relu(x):
    return np.maximum(x, 0)


def fcann2_train(X, Y_):
    '''
      Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array NxC

      Povratne vrijednosti
        w, b: parametri logističke regresije
    '''
    N, D = X.shape
    C = np.amax(Y_) + 1
    H = 5
    W1 = np.random.randn(D, H)
    b1 = np.zeros(H)
    W2 = np.random.randn(H, C)
    b2 = np.zeros(C)
    Yoh = data.class_to_onehot(Y_)

    for i in range(param_niter):
        # eksponencirane klasifikacijske mjerestade
        # pri računanju softmaksa obratite pažnju
        # na odjeljak 4.1 udžbenika
        # (Deep Learning, Goodfellow et al)!
        # scores = np.add(np.dot(X,W),b)  # N x C
        s1 = np.add(np.dot(X, W1), b1)
        h1 = relu(s1)
        s2 = np.add(np.dot(h1, W2), b2)
        # probs = stable_softmax(s2)

        expscores = np.exp(s2)  # N x C

        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1).reshape((N, 1))  # N x 1

        # logaritmirane vjerojatnosti razreda
        probs = np.divide(expscores, sumexp)  # stable_softmax(scores) # N x C

        # probs = stable_softmax(s2)

        logprobs = np.log(probs)  # N x C

        # Yoh = data.class_to_onehot(Y_)
        # gubitak
        # loss = (-np.sum(np.log(np.power(probs, Yoh)) * np.power((1-probs), (1-Yoh))))  # scalar

        loss = -np.sum(logprobs)
        loss += param_lambda * (np.linalg.norm(W2, 2) + np.linalg.norm(W1, 2))

        loss /= X.shape[0]

        # dijagnostički ispis
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))
            # Y = np.argmax(probs, axis=1)
            # rect = (np.min(X, axis=0), np.max(X, axis=0))
            # data.graph_surface(lambda x: np.argmax(fcann2_classify(x, W1, b1, W2, b2), axis=1), rect, offset=0.5)
            #
            # # graph the data points
            # data.graph_data(X, Y_, Y, special=[])
            #
            # data.plt.show()

        # derivacije komponenata gubitka po mjerama
        dL_ds2 = probs - Yoh  # N x C

        # gradijenti parametara
        # grad_W = np.dot(dL_ds.T, X) / N  # C x D (ili D x C)
        # grad_b = np.average(dL_ds, axis=0) # C x 1 (ili 1 x C)

        grad_W2 = np.dot(dL_ds2.T, h1)
        grad_W2 /= N
        grad_W2 += param_lambda * W2.T
        grad_b2 = np.average(dL_ds2, axis=0)

        dl_ds1 = np.dot(dL_ds2, W2.T) * np.greater(s1, 0).astype(int)

        grad_W1 = np.dot(dl_ds1.T, X)
        grad_W1 /= N
        grad_W1 += param_lambda * W1.T
        grad_b1 = np.average(dl_ds1, axis=0)

        # poboljšani parametri
        # W += -get_param_delta(i) * grad_W.T
        # b += -get_param_delta(i) * grad_b

        # W2 += -get_param_delta(i) * grad_W2.T
        # W1 += -get_param_delta(i) * grad_W1.T
        # b2 += -get_param_delta(i) * grad_b2
        # b1 += -get_param_delta(i) * grad_b1

        W2 += -param_delta * grad_W2.T
        W1 += -param_delta * grad_W1.T
        b2 += -param_delta * grad_b2
        b1 += -param_delta * grad_b1

    return W1, b1, W2, b2


def fcann2_classify(X, W1, b1, W2, b2):
    s1 = np.add(np.dot(X, W1), b1)
    h1 = relu(s1)
    s2 = np.add(np.dot(h1, W2), b2)
    return stable_softmax(s2)


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    # train the model
    W1, b1, W2, b2 = fcann2_train(X, Y_)

    # evaluate the model on the training dataset
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    # AP = data.eval_AP(Y_[probs.argsort()])
    # print(accuracy, recall, precision, AP)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: np.argmax(fcann2_classify(x, W1, b1, W2, b2), axis=1), rect, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    data.plt.show()
