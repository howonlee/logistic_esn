import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # state = npr.random(100) - 0.5
    state = npr.random(100)
    esn = npr.random(size=(100,100)) - 0.5
    # esn[np.invert(np.eye(100,100, dtype=np.bool))] = 0
    data = np.loadtxt('MackeyGlass_t17.txt')
    spec = np.max(np.abs(npl.eig(esn)[0]))
    esn /= spec
    esn *= 1.3
    activation = []
    for x in xrange(10000):
        # state = np.tanh(np.dot(state, esn))
        state = 4 * np.multiply(state, 1 - state)
        activation.append(state[2])
    plt.scatter(activation[:-1], activation[1:], alpha=0.2)
    plt.show()
