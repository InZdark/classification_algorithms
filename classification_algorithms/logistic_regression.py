import numpy as np
import matplotlib.pyplot as plt

''' Assistant Functions '''

def sigmoid(h, epsilon = 1e-5):
    return 1/(1 + np.exp(-h + epsilon))

def cross_entropy(y, p_hat, epsilon = 1e-3):
    return -(1/len(y)) * np.sum(y * np.log(p_hat + epsilon)\
                                + (1- y) * np.log(1 - p_hat + epsilon))

class LogisticRegression():

    def __init__(self, size):
        ''' Initialize weights and bias based on the number of features. '''
        self.w = np.random.randn(size)
        self.b = np.random.randn(1)

    def fit(self, X_trn, y_trn, X_val, y_val, lr = 1e-1, epochs = 1e3, show_curve = False):

        epochs = int(epochs)
        print_pt = epochs//5
        N, D = X_trn.shape

        J_trn = np.zeros(epochs) # train loss
        J_val = np.zeros(epochs) # validation loss

        for epoch in range(epochs): # start to train
            # calculate probability
            p_hat = self.__forward(X_trn)
            # save training process
            J_trn[epoch] = cross_entropy(y_trn, p_hat)
            J_val[epoch] = cross_entropy(y_val, self.__forward(X_val))
            # weights update
            self.w -= lr*(1/N)*X_trn.T@(p_hat - y_trn)
            self.b -= lr*(1/N)*np.sum(p_hat - y_trn)
            # print progress
            if epoch % print_pt == 0:
                print('Epoch: {}, train error: {:.4f}, valid error: {:.4f}'.\
                      format(epoch, J_trn[epoch], J_val[epoch]))
        # plot curve
        if show_curve:
            plt.figure(figsize = (15, 6))
            # train plot
            plt.subplot(121); plt.plot(J_trn)
            plt.xlabel('epochs'); plt.ylabel('$\mathcal{J}$')
            plt.title('Training Curve', fontsize = 15)
            # valid plot
            plt.subplot(122); plt.plot(J_val)
            plt.xlabel('epochs'); plt.ylabel('$\mathcal{J}$')
            plt.title('Validation Curve', fontsize = 15)
        # return training process
        return {'J_trn': J_trn, 'J_val': J_val}

    def __forward(self, X):
        return sigmoid(X@self.w + self.b)

    def predict(self, X, thresh = 0.5):
        return (self.__forward(X) >= thresh).astype(np.int32)
