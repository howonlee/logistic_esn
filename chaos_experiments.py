import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import matplotlib.pyplot as plt
import time

def saturate(val):
    val[val < 0] = 0
    val[val > 1] = 1
    return val

def test_esn(data):
    for _ in xrange(10):
        state = npr.random(100) - 0.5
        esn = npr.random(size=(100,100)) - 0.5
        w_in = npr.random(size=(100, 1)) - 0.5
        spec = np.max(np.abs(npl.eig(esn)[0]))
        esn /= spec
        esn *= 1.3
        activation = []
        for x in xrange(2000):
            state = np.tanh(np.dot(state, esn) + np.dot(w_in, data[x]))
            if x % 500 == 0:
                print x
            if x > 500:
                activation.append(state[3])
        plt.close()
        plt.scatter(activation[:-1], activation[1:], alpha=0.2)
        plt.show()

def test_logistic(data):
    for _ in xrange(10):
        npr.seed(int(time.time()))
        fold_in_data = lambda state, data: saturate(state + 2 * data)
        state = npr.random(100)
        w_in = npr.random(size=(100, 1)) - 0.5
        activation = []
        for x in xrange(2500):
            state = 5 * np.multiply(fold_in_data(state, np.dot(w_in, data[x])), 1 - fold_in_data(state, np.dot(w_in, data[x])))
            if x % 500 == 0:
                print x
            if x > 500:
                activation.append(state[2])
        plt.close()
        plt.scatter(activation[:-1], activation[1:], alpha=0.2)
        plt.show()

if __name__ == "__main__":
    data = np.loadtxt('MackeyGlass_t17.txt')
    # test_logistic(data)
    test_esn(data)
