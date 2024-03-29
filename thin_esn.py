import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.sparse as sci_sp
import scipy.stats as sci_st
import datetime
import math
import tensorflow as tf
import sklearn
import sklearn.linear_model

class ESN:
    def __init__(self, in_size, out_size, res_size, reduction_size, a, spectral_radius):
        self.in_size = in_size
        self.reduction_size = reduction_size
        self.out_size = out_size
        self.res_size = res_size
        self.a = a
        self.Win = (npr.rand(self.res_size,1 + self.in_size)-0.5) * 1
        # * 2 to have it be in the range [-1, 1]
        # self.W = (npr.rand(self.res_size, 1) - 0.5) * 2 * spectral_radius
        self.W = (npr.rand(self.res_size, self.res_size) - 0.5) * 2
        self.W /= np.sqrt(self.res_size)
        self.W *= spectral_radius
        self.W2 = npr.normal(size=(self.reduction_size, self.res_size)) # pay your taxes, kids
        self.W2 /= np.sqrt(self.res_size)
        self.W2 *= spectral_radius
        self.nonlinear = np.tanh
        self.activation_function = self.run_esn_activation
        self.setup_net()
        assert net in self
        # self.activation_function = self.run_thin_activation
        self.Wout = None #untrained as of yet

    def setup_net(self):
        X = tf.placeholder('float', [None, self.res_size])
        y = tf.placeholder('float', [None, self.out_size])
        Wout = tf.Variable(tf.random_normal([self.res_size, self.out_size]) * 0.1)
        activation = tf.matmul(X, Wout)
        cost = tf.reduce_sum(tf.pow(activation-y, 2)) / (2 * n_samples)
        # eventually, poke about for a NAG optimizer
        optimizer = tf.train.MomentumOptimizer(0.001, 0.999).minimize(cost)
        self.net = {
            'X': X,
            'y': y,
            'Wout': Wout,
            'b': b,
            'activation': activation,
            'cost': cost,
            'optimizer': optimizer
        }

    def run_esn_activation(self, prev_activation, datum):
        biased_data = np.atleast_2d(np.hstack((1, datum))).T
        internal_signal = self.W.dot(prev_activation)
        return (1-self.a) * prev_activation + \
               self.a * self.nonlinear(np.dot(self.Win, biased_data) + internal_signal)

    def run_thin_activation(self, prev_activation, datum):
        biased_data = np.atleast_2d(np.hstack((1, datum))).T
        internal_signal = np.multiply(self.W, prev_activation)
        return (1-self.a) * prev_activation + \
               self.a * self.nonlinear(np.dot(self.Win, biased_data) + internal_signal)

    def run_reservoir(self, data, init_len):
        train_len = len(data)
        res = np.zeros((1+self.in_size+self.res_size, train_len-init_len))
        x = np.zeros((self.res_size,1))
        for t in range(len(data)):
            if t % 1000 == 0:
                print "running: ", t, " / ", len(data), datetime.datetime.now()
            u = data[t]
            x = self.activation_function(x, u)
            if t >= init_len:
                res[:, t-init_len] = np.hstack((np.atleast_2d(1), np.atleast_2d(u), x.T))[0,:]
        return res, x

    def run_thin(self, data, init_len):
        train_len = len(data)
        res = np.zeros((1+self.in_size+self.reduction_size, train_len-init_len))
        x = np.zeros((self.res_size,1))
        for t in range(len(data)):
            if t % 1000 == 0:
                print "running: ", t, " / ", len(data), datetime.datetime.now()
            u = data[t]
            x = self.activation_function(x, u)
            # and then do the reduction, too, here
            x2 = self.W2.dot(x)
            if t >= init_len:
                res[:, t-init_len] = np.hstack((np.atleast_2d(1), np.atleast_2d(u), x2.T))[0,:]
        return res, x

    def train(self, res, data, reg):
        # ridge regression
        print "begin training..."
        print "first dot shape: ", np.dot(data.T, res.T).shape
        print "second dot shape: ", np.dot(res, res.T).shape
        self.Wout = np.dot(
                np.dot(data.T, res.T),
                npl.inv(
                    np.dot(res, res.T) +\
                    reg * np.eye(1 + self.in_size + self.res_size)
                )
            )
        print "finished training..."

    def train_thin(self, res, data, reg):
        print "begin training..."
        self.Wout = np.dot(
                np.dot(data.T, res.T),
                npl.inv(
                    np.dot(res, res.T) +\
                    reg * np.eye(1 + self.in_size + self.reduction_size)
                )
            )
        print "finished training..."

    def train_sgd(self, res, data):
        print "begin training sgd..."
        # regressor = sklearn.linear_model.SGDRegressor(alpha=1e-7, n_iter=4000, shuffle=False, verbose=1, eta0=0.00001, learning_rate='constant')
        regressor = sklearn.linear_model.SGDRegressor(alpha=1e-5, n_iter=10000, shuffle=False, verbose=1)
        regressor.fit(res.T, data)
        self.Wout = regressor.coef_
        print "finished training..."

    def train_sgd_ridge(self, res, data):
        print "begin training sgd ridge..."
        regressor = sklearn.linear_model.RidgeCV()
        regressor.fit(res.T, data)
        self.Wout = regressor.coef_
        print "finished training..."

    def train_sgd_momentum(self, res, data, num_epochs=20):
        with tf.Session() as sess:
            tf.initialize_all_variables()
            for epoch in num_epochs:
                for idx, res_datum in enumerate(res): # hope this works
                    out_datum = data[idx]
#################
#################
#################

    def generate(self, init_u, init_x, test_len):
        u, x = init_u, init_x
        Y = np.zeros((self.out_size, test_len))
        for t in xrange(test_len):
            if t % 1000 == 0:
                print "generating: ", t, " / ", test_len, datetime.datetime.now()
            x = self.activation_function(x, u)
            y = np.dot(self.Wout, np.hstack((np.atleast_2d(1), np.atleast_2d(u), x.T)).T)
            y = np.atleast_2d(y)
            Y[:, t] = y[:, 0]
            u = y[:, 0]
        return Y

    def generate_thin(self, init_u, init_x, test_len):
        u, x = init_u, init_x
        Y = np.zeros((self.out_size, test_len))
        for t in xrange(test_len):
            if t % 1000 == 0:
                print "generating: ", t, " / ", test_len, datetime.datetime.now()
            x = self.activation_function(x, u)
            x2 = self.W2.dot(x)
            y = np.dot(self.Wout, np.hstack((np.atleast_2d(1), np.atleast_2d(u), x2.T)).T)
            y = np.atleast_2d(y)
            Y[:, t] = y[:, 0]
            u = y[:, 0]
        return Y

    def generate_nn(self, init_u, init_x, test_len):
        pass
#######################
#######################
#######################

    def predict(self, init_u, init_x, data, test_len, train_len):
        u, x = init_u, init_x
        Y = np.zeros((self.out_size, test_len))
        for t in xrange(test_len):
            if t % 1000 == 0:
                print "generating: ", t, " / ", test_len, datetime.datetime.now()
            x = self.activation_function(x, u)
            y = np.dot(self.Wout, np.hstack((np.atleast_2d(1), np.atleast_2d(u), x.T)).T)
            y = np.atleast_2d(y)
            Y[:, t] = y[:, 0]
            u = data[train_len + t + 1]
        return Y

    def predict_thin(self, init_u, init_x, data, test_len, train_len):
        u, x = init_u, init_x
        Y = np.zeros((self.out_size, test_len))
        for t in xrange(test_len):
            if t % 1000 == 0:
                print "generating: ", t, " / ", test_len, datetime.datetime.now()
            x = self.activation_function(x, u)
            x2 = self.W2.dot(x)
            y = np.dot(self.Wout, np.hstack((np.atleast_2d(1), np.atleast_2d(u), x2.T)).T)
            y = np.atleast_2d(y)
            Y[:, t] = y[:, 0]
            u = data[train_len + t + 1]
        return Y

    def get_spectral_radius(self):
        assert isinstance(self.W, np.ndarray) # so no sparse matrices!
        return np.max(np.abs(npl.eig(self.W)[0]))

    spectral_radius = property(get_spectral_radius)

def mse(data, pred, error_len):
    return np.sum(np.square(data.ravel()[:error_len] - pred.ravel()[:error_len])) / error_len

if __name__ == "__main__":
    data = np.loadtxt('MackeyGlass_t17.txt')
    train_length = 5000
    burnin_length = 1000
    test_length = 300
    error_length = 300
    num_total_iters = 5
    reg = 1e-7

    train_data = data[:train_length]
    train_target = data[burnin_length+1:train_length+1]
    test_data = data[train_length+1:train_length+test_length+1]

    num_successes = 0
    ses = []

    for curr_iter in xrange(num_total_iters):
        try:
            print "current iter: ", curr_iter
            print "start creating net..."
            net = ESN(
                    in_size=1,
                    out_size=1,
                    res_size=1000,
                    reduction_size=1000,
                    a=1.0,
                    spectral_radius=0.9)
            print "finished creating net..."
            # res, x = net.run_thin(data=train_data, init_len=burnin_length)
            res, x = net.run_reservoir(data=train_data, init_len=burnin_length)
            outer = np.cov(res)
            eigs = npl.eig(outer)[0]
            print np.max(np.abs(eigs))
            print np.min(np.abs(eigs))
            # net.train_thin(res=res, data=train_target, reg=reg)
            # net.train_sgd(res=res, data=train_target)
            # net.train_sgd_ridge(res=res, data=train_target)
            net.train_sgd_momentum(res=res, data=train_target)
            out = net.generate(data[train_length], x, test_length)
            # out = net.generate_thin(data[train_length], x, test_length)
            # out = net.predict_thin(data[train_length], x, data, test_length, train_length)
            plt.close()
            plt.plot(out.T)
            plt.plot(test_data)
            plt.show()
            ses.append(mse(test_data, out, error_length))
            if mse(test_data, out, error_length) < 0.001:
                num_successes += 1
        except OverflowError, ValueError:
            print "iter failed"
    print num_successes, " successes!"
    print ses
