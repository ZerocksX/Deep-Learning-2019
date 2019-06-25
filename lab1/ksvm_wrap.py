from sklearn import svm
from faks.du.lab1 import data
import numpy as np


class KSVMWrap(object):
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto', kernel='rbf'):
        """
        Konstruira omotač i uči RBF SVM klasifikator
        X,Y_:            podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre
        """
        self.X = X
        self.Y_ = Y_
        self.param_svm_c = param_svm_c
        self.param_svm_gamma = param_svm_gamma
        self.clf = svm.SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True, decision_function_shape='ovo', kernel=kernel)
        self.clf.fit(self.X, self.Y_)

    def predict(self, X):
        """
        Predviđa i vraća indekse razreda podataka X
        """
        return self.clf.predict(X)

    def get_scores(self, X):
        """
        Vraća klasifikacijske mjere
        (engl. classification scores) podataka X;
        ovo će vam trebati za računanje prosječne preciznosti.
        """
        return self.clf.predict_proba(X)

    def suport(self):
        """
        Indeksi podataka koji su odabrani za potporne vektore
        """
        return self.clf.support_


if __name__ == '__main__':
    np.random.seed(100)
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    svm = KSVMWrap(X, Y_)
    Y = svm.predict(X)
    probs = svm.get_scores(X)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print('acc', accuracy)
    print('recall', recall)
    print('precision', precision)

    # iscrtaj rezultate, decizijsku plohu

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: np.array([-p[0] if p[0] > 0.5 else p[1] for p in svm.get_scores(x)]), rect, offset=0)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[svm.suport()])

    data.plt.show()
