import numpy as np

def linear(H):
    return H

def ReLU(H):
    return H * (H > 0)

def sigmoid(H):
    return 1/(1 + np.exp(-H))

def tanh(H):
    return np.tanh(H)

def softmax(H):
    eH = np.exp(H)
    return eH/eH.sum(axis = 1, keepdims = True)

def cross_entropy(Y, P_hat):
    return -(1/len(Y)) * np.sum(Y * np.log(P_hat))

def OLS(Y, Y_hat):
    return (1/(2*len(Y))) * np.sum((Y - Y_hat)**2)

def derivatives(Z, a):
    if a == linear:
        return 1
    elif a == sigmoid:
        return Z * (1 - Z)
    elif a == tanh:
        return 1 - Z * Z
    elif a == ReLU:
        return (Z > 0).astype(int)
    else:
        ValuError('Unknown Activation')
        
def one_hot_encode(y):
    N = len(y)      # number of samples
    K = len(set(y)) # number of categories
    Y = np.zeros((N, K))
    for i in range(N):
        Y[i, y[i]] = 1
    return Y

def accuracy(y, y_hat):
    return np.mean(y == y_hat)
    
class MLPClassifier():
    
    def __init__(self, input_dims, output_dims, architecture, activations = None):
        
        self.architecture = architecture
        self.L = len(architecture) + 1    # number of layers, include input layer
        self.activations = activations
        self.W = {l: np.random.randn(M[0], M[1]) for \
                  l, M in enumerate(zip(([input_dims] + self.architecture), (self.architecture + [output_dims])), 1)}
        self.b = {l: np.random.randn(M) for l, M in enumerate(self.architecture + [output_dims], 1)}
        
    def fit(self, X, y, X_val, y_val, lr = 1e-1, epochs = 1000, show_curve = False):
        
        epochs = int(epochs)
        
        Y = one_hot_encode(y.astype(int))
        Y_val = one_hot_encode(y_val.astype(int))
        
        N, D = X.shape # input  layer units: D
        K = Y.shape[1] # output layer units: K
        
        if self.activations is None:
            self.a = {l: ReLU for l in range(1, self.L)}
        else:
            self.a = {l: act for l, act in enumerate(self.activations, 1)}
        # classification
        self.a[self.L] = softmax
        J = np.zeros(epochs)
        
        for epoch in range(epochs):
            
            self.forward(X)
            # save losses
            J[epoch] = cross_entropy(Y, self.Z[self.L])
            # weights update
            dH = (1/N) * (self.Z[self.L] - Y)
            for l in sorted(self.W.keys(), reverse = True):
                dW = self.Z[l - 1].T@dH
                db = dH.sum(axis = 0)
                self.W[l] -= lr*dW
                self.b[l] -= lr*db
                
                if l > 1:
                    dZ = dH@self.W[l].T
                    dH = dZ*derivatives(self.Z[l - 1], self.a[l - 1])
            if (epoch+1) % 500 == 0:
                self.forward(X_val)
                J_val = cross_entropy(Y_val, self.Z[self.L])
                print('train error: {:.4f}, train accuracy: {:.3f}'.format(J[epoch],
                                                                           accuracy(y, self.predict(X))))
                print('valid error: {:.4f}, valid accuracy: {:.3f}'.format(J_val,
                                                                           accuracy(y_val, self.predict(X_val))))
                print('------------------------------------------------------------------')
                    
        if show_curve:
            plt.figure(); plt.plot(J)
            plt.xlabel('epochs'); plt.ylabel('$\mathcal{J}$')
            plt.show()
        
    def forward(self, X):
        self.Z = {0: X}
        
        for l in sorted(self.W.keys()):
            self.Z[l] = self.a[l](self.Z[l - 1]@self.W[l] + self.b[l])
    
    def predict(self, X):
        self.forward(X)
        return self.Z[self.L].argmax(axis = 1)
