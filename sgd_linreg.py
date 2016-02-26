import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

### implement OLS with SGD only
### we actually really need this to be truly online

def generate_x(ys):
    xs = np.zeros((len(ys), 3))
    for idx, y in enumerate(ys):
        new_x = y * 2
        xs[idx] = new_x
    xs += npr.random() * 0.01
    return xs


if __name__ == "__main__":
    y = npr.normal(size=(20000,3))
    X = generate_x(y)
    params = npr.normal(size=(3,3))
    energies = []

    # begin gradient descent
    for idx, datum in enumerate(X):
        res = datum.dot(params)
        delta = (res - y[idx])
        energies.append(np.sum(delta ** 2))
        diff = datum.T.dot(delta)
        params -= 0.0001 * diff
    plt.plot(energies)
    plt.show()
