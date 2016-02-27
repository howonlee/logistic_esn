import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import matplotlib.pyplot as plt
import time

def saturate(val):
    val[val < 0] = 0
    val[val > 1] = 1
    return val

if __name__ == "__main__":
    # state = npr.random(100) - 0.5
    data = np.loadtxt('MackeyGlass_t17.txt')
    # for _ in xrange(10):
    #     npr.seed(int(time.time()))
    #     fold_in_data = lambda state, data: saturate(state + 0.01*data)
    #     state = npr.random(100)
    #     activation = []
    #     for x in xrange(10000):
    #         # state = np.tanh(np.dot(state, esn))
    #         state = 2.5 * np.multiply(fold_in_data(state, data[x]), 1 - fold_in_data(state, data[x]))
    #         if x > 2000:
    #             activation.append(state[2])
    #     plt.close()
    #     plt.scatter(activation[:-1], activation[1:], alpha=0.2)
    #     plt.show()
    for _ in xrange(10):
        state = npr.random(100)
        esn = npr.random(size=(100,100)) - 0.5
        spec = np.max(np.abs(npl.eig(esn)[0]))
        esn /= spec
        esn *= 1.3
        activation = []
        for x in xrange(10000):
            state = np.tanh(np.dot(state, esn) + data[x])
            if x > 2000:
                activation.append(state[3])
        plt.close()
        plt.scatter(activation[:-1], activation[1:], alpha=0.2)
        plt.show()
