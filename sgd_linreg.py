import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

### implement OLS with SGD only
### we actually really need this to be truly online

def generate_x(ys):
    xs = np.zeros((len(ys), 12))
    for idx, y in enumerate(ys):
        new_x = np.hstack((y, y*2, y - 300, y * 100))
        xs[idx] = new_x
    xs += npr.random() * 0.01
    return xs


if __name__ == "__main__":
    y = npr.normal(size=(300,3))
    X = generate_x(y)
    params = npr.normal(size=(12,3))
    print params
    energies = []

    # begin gradient descent
    for idx, datum in enumerate(X):
        res = datum.dot(params)
        delta = res - y[idx]
        energy = np.sum(delta ** 2)
        print "energy: ", energy
        energies.append(energy)
        diff = np.outer(delta, datum).T
        # print "diff: ", diff
        params -= 0.0005 * diff
    plt.plot(energies)
    plt.show()
