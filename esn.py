import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_hiddens = 100
    num_ins, num_outs = 1, 1
    data = np.loadtxt("MackeyGlass_t17.txt")[:8000]
    train, test = data[:1000], data[1000:]
    curr_hidden = npr.random(num_hiddens) - 0.5
    weights_ir = npr.random((num_ins, num_hiddens)) - 0.5
    weights_rr = npr.random((num_hiddens, num_hiddens)) - 0.5
    weights_ro = npr.random((num_hiddens, num_outs)) - 0.5 # there get learned
    # try the activations, basically
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
