import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.sparse as sci_sp
import scipy.stats as sci_st
import datetime
import math
import tensorflow as tf

class ESN:
    def __init__(self, in_size, out_size, res_size, a, spectral_radius):
        self.in_size = in_size
        self.out_size = out_size
        self.res_size = res_size
        self.a = a
        self.Win = (npr.rand(self.res_size,1 + self.in_size)-0.5) * 1
        self.W = npr.rand(self.res_size, self.res_size) - 0.5
        self.nonlinear = np.tanh
        self.mlp = {}
        self.tf_sess = tf.Session()
        self.setup_mlp()
        self.mlp_trained = False
        # relu:
        # self.nonlinear = lambda x: np.maximum(x, 0)
        self.W *= spectral_radius / self.spectral_radius
        # no Wout: the MLP reads it out

    def setup_mlp(self):
        self.mlp["keep_prob"] = tf.placeholder(tf.float32)
        self.mlp["inputs"] = tf.placeholder(tf.float32, shape=[None, self.res_size + 1], name="X")
        # self.mlp["inputs"] = tf.nn.dropout(self.mlp["inputs"], self.mlp["keep_prob"])
        self.mlp["outputs"] = tf.placeholder(tf.float32, shape=[None, self.out_size], name="Y")
        xavier_io = math.sqrt(6.0 / (self.res_size + self.in_size + 1 + self.out_size))
        # bias is added in beforehands, that's the +1
        self.mlp["w_io"] = tf.Variable(tf.random_uniform([self.res_size + 1, self.out_size],
            minval=-xavier_io, maxval=xavier_io, dtype=tf.float32))
        self.mlp["out"] = tf.matmul(self.mlp["inputs"], self.mlp["w_io"])
        self.mlp["loss"] = tf.nn.l2_loss(self.mlp["out"] - self.mlp["outputs"])
        self.mlp["global_step"] = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.05, self.mlp["global_step"],
                                                   1000, 0.99, staircase=True)
        self.mlp["train"] = tf.train.AdamOptimizer(learning_rate).minimize(self.mlp["loss"], global_step=self.mlp["global_step"])
        init = tf.initialize_all_variables()
        self.tf_sess.run(init)

    def run_activation(self, prev_activation, datum):
        biased_data = np.atleast_2d(np.hstack((1, datum))).T
        internal_signal = self.W.dot(prev_activation)
        return (1-self.a) * prev_activation + \
               self.a * self.nonlinear(np.dot(self.Win, biased_data) + internal_signal)

    def mlp_train(self, data, init_len):
        # use only dropout
        num_epochs = 5
        print "begin training MLP..."
        for epoch in xrange(num_epochs):
            print epoch, " / ", num_epochs
            data_idx = 0
            # begin burnin
            curr_x = npr.random(size=(self.res_size,1)) * 0.1
            for x_idx in xrange(init_len):
                curr_x = self.run_activation(curr_x, data[data_idx])
                data_idx += 1
            # finish burnin
            # begin actual training
            for start, end in zip(range(data_idx, len(data)-1, 128), range(data_idx+128, len(data)-1, 128)):
                curr_res = np.zeros((127, 1+self.res_size))
                for res_idx in xrange((end - 1) - start):
                    curr_x = self.run_activation(curr_x, data[data_idx])
                    curr_res[res_idx, :] = np.vstack((np.atleast_2d(1), curr_x))[:, 0]
                    data_idx += 1
                outs = np.atleast_2d(data[data_idx-127:data_idx]).T
                self.tf_sess.run(self.mlp["train"],
                        feed_dict={
                            self.mlp["inputs"]: curr_res,
                            self.mlp["outputs"]: outs,
                            self.mlp["keep_prob"]: 0.9
                            })
            # finish actual training
        self.mlp_trained = True
        print "finished training MLP..."
        return curr_x

    def mlp_generate(self, init_u, init_x, test_len):
        assert self.mlp_trained
        u, x = init_u, init_x
        Y = np.zeros((self.out_size, test_len))
        for t in xrange(test_len):
            if t % 1000 == 0:
                print "generating: ", t, " / ", test_len, datetime.datetime.now()
            x = self.run_activation(x, u)
            curr_res = np.vstack((1, x)).T
            y = np.atleast_2d(self.mlp["out"].eval(session=self.tf_sess,
                              feed_dict={
                                  self.mlp["inputs"]: curr_res,
                                  self.mlp["outputs"]: np.zeros((1, self.out_size)),
                                  self.mlp["keep_prob"]: 1.0}))
            Y[:, t] = y.T[:, 0]
            u = y.T[:, 0]
        return Y

    def mlp_predict(self, init_u, init_x, data, test_len, train_len):
        assert self.mlp_trained
        u, x = init_u, init_x
        Y = np.zeros((self.out_size, test_len))
        for t in xrange(test_len):
            if t % 1000 == 0:
                print "generating: ", t, " / ", test_len, datetime.datetime.now()
            x = self.run_activation(x, u)
            curr_res = np.vstack((1, x)).T
            y = np.atleast_2d(self.mlp["out"].eval(session=self.tf_sess,
                              feed_dict={
                                  self.mlp["inputs"]: curr_res,
                                  self.mlp["outputs"]: np.zeros((1, self.out_size)),
                                  self.mlp["keep_prob"]: 1.0}))
            Y[:, t] = y.T[:, 0]
            u = data[train_len + t + 1]
        return Y

    def get_spectral_radius(self):
        return np.max(np.abs(npl.eig(self.W)[0]))

    spectral_radius = property(get_spectral_radius)

def mse(data, pred, error_len):
    return np.sum(np.square(data.ravel()[:error_len] - pred.ravel()[:error_len])) / error_len


if __name__ == "__main__":
    data = np.loadtxt('MackeyGlass_t17.txt')
    data -= np.min(data) #positivify it
    data /= np.max(data) #normalize
    train_length = 5000
    burnin_length = 1000
    test_length = 3000
    error_length = 3000
    num_total_iters = 5

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
                    res_size=300,
                    a=0.3,
                    spectral_radius=1.25)
            print "finished creating net..."
            x = net.mlp_train(data=train_data, init_len=burnin_length)
            # out = net.mlp_predict(data[train_length], x, data, test_length, train_length)
            out = net.mlp_generate(data[train_length], x, test_length)
            print out
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
