import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt("MackeyGlass_t17.txt")[:8000]
    train, test = data[:1000], data[1000:]
    plt.plot(test)
    plt.show()
    # for n in xrange(2,100):
    #     time = 10000
    #     curr_hidden = npr.random(n) - 0.5
    #     weights = npr.random((n,n)) - 0.5
    #     activation_progression = []
    #     activation_progression_2 = []
    #     for x in xrange(time):
    #         curr_hidden = np.tanh(np.dot(weights, curr_hidden))
    #         activation_progression.append(curr_hidden[0])
    #         activation_progression_2.append(curr_hidden[1])
    #     plt.close()
    #     plt.scatter(activation_progression, activation_progression_2)
    #     plt.show()
