import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# plain esn

curr_hidden = npr.random(100) - 0.5
weights = npr.random((100,100)) - 0.5

if __name__ == "__main__":
    time = 1000
    activation_progression = []
    activation_progression_2 = []
    for x in xrange(time):
        curr_hidden = np.tanh(np.dot(weights, curr_hidden))
        activation_progression.append(curr_hidden[0])
        activation_progression_2.append(curr_hidden[1])
    plt.scatter(activation_progression, activation_progression_2)
    plt.show()
