from lab0 import data
import numpy as np

param_niter = 10000
delta_big = 1e-1
delta_small = 1e-4
get_param_delta = lambda iteration: (iteration / param_niter) * delta_small + ((param_niter - iteration) / param_niter) * delta_big
# get_param_delta = lambda x : delta_big
# stabilni softmax
def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs

def logreg_train(X, Y_):
    '''
      Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array NxC

      Povratne vrijednosti
        w, b: parametri logističke regresije
    '''
    N, D = X.shape
    C = np.amax(Y_) + 1
    W = np.random.randn(D, C)
    b = np.zeros(C)

    for i in range(param_niter):
        # eksponencirane klasifikacijske mjerestade
        # pri računanju softmaksa obratite pažnju
        # na odjeljak 4.1 udžbenika
        # (Deep Learning, Goodfellow et al)!
        scores = np.add(np.dot(X,W),b)  # N x C
        expscores = np.exp(scores)  # N x C

        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1).reshape((N, 1)) # N x 1

        # logaritmirane vjerojatnosti razreda
        probs = np.divide(expscores, sumexp) # stable_softmax(scores) # N x C

        # probs = stable_softmax(scores)

        logprobs = np.log(probs)  # N x C

        Yoh = data.class_to_onehot(Y_)
        # gubitak
        # loss = (-np.sum(np.log(np.power(probs, Yoh)) * np.power((1-probs), (1-Yoh))))  # scalar

        loss = -np.sum(Yoh * logprobs)

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - Yoh  # N x C

        # gradijenti parametara
        grad_W = np.dot(dL_ds.T, X) / N  # C x D (ili D x C)
        grad_b = np.average(dL_ds, axis=0) # C x 1 (ili 1 x C)

        # poboljšani parametri
        W += -get_param_delta(i) * grad_W.T
        b += -get_param_delta(i) * grad_b

    return W, b


def logreg_classify(X, W, b):
    scores = np.add(np.dot(X,W),b)  # N x C
    return stable_softmax(scores)


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 100)

    # train the model
    W, b = logreg_train(X, Y_)


    # evaluate the model on the training dataset
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    print(accuracy, recall, precision)
    # AP = data.eval_AP(Y_[probs.argsort()])

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: np.argmax(logreg_classify(x, W, b), axis=1), rect, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    data.plt.show()

