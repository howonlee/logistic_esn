import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def compute_activations():
    activations = np.zeros(something)
    activations = np.tanh(np.dot(weights, curr_hidden))
    return activations

def run_network():
    pass

def train_out_weights():
    pass

def test():
    pass

if __name__ == "__main__":
    num_hiddens = 100
    num_ins, num_outs = 1, 1
    data = np.loadtxt("MackeyGlass_t17.txt")[:8000]
    train, test = data[:1000], data[1000:]
    curr_hidden = npr.random(num_hiddens) - 0.5
    weights_ir = npr.random((num_ins, num_hiddens)) - 0.5
    weights_rr = npr.random((num_hiddens, num_hiddens)) - 0.5
    weights_ro = npr.random((num_hiddens, num_outs)) - 0.5 # these get learned
