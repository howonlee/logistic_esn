import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

### implement OLS with SGD only
### we actually really need this to be truly online


if __name__ == "__main__":
    X = npr.random(size=(20000,10))
    y = npr.normal(size=(20000,3))
    params = npr.normal(size=(10,3))
    print params

    # begin gradient descent
    for idx, datum in enumerate(X):
        print "datum: ", datum
        print "res: ", datum.dot(params)
        print "y: ", y[idx]
