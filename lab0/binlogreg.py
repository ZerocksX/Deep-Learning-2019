import numpy as np
from lab0 import data

param_niter = 10000
get_param_delta = lambda iteration: (iteration / param_niter) * 1e-4 + ((param_niter - iteration) / param_niter) * 1e-1

def sigmoid(x):
    return np.exp(x) / ( 1 + np.exp(x))


def binlogreg_train(X, Y_):
    '''
      Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

      Povratne vrijednosti
        w, b: parametri logističke regresije
    '''
    N, D = X.shape
    w = np.random.randn(D)
    b = 0

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):
        # klasifikacijske mjere
        scores = np.dot(X, w) + b  # N x 1

        # vjerojatnosti razreda c_1
        probs = sigmoid(scores)  # N x 1

        # gubitak
        loss = - np.sum(np.log(np.power(probs, Y_) * ( np.power((1-probs), (1-Y_)))))  # scalar

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - Y_  # N x 1

        # gradijenti parametara
        grad_w = np.average(dL_dscores * X.T, axis=(1,)).T  # D x 1
        grad_b = np.average(dL_dscores)  # 1 x 1

        # poboljšani parametri
        w += -get_param_delta(i) * grad_w
        b += -get_param_delta(i) * grad_b

        rect = (np.min(X, axis=0), np.max(X, axis=0))
        data.graph_surface(lambda x: binlogreg_classify(x, w, b), rect, offset=0.5)
        data.plt.show()

    return w, b


def binlogreg_classify(X, w, b):
    '''
      Argumenti
          X:    podatci, np.array NxD
          w, b: parametri logističke regresije

      Povratne vrijednosti
          probs: vjerojatnosti razreda c1
    '''
    return sigmoid(np.dot(X,w) + b)


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)


    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = (probs > 0.5).astype(int)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: binlogreg_classify(x, w, b), rect, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    data.plt.show()
