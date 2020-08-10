import numpy as np

class KNNClassifier():
    def fit(self, X, y):
        self.X = X
        self.y = y.astype(int)
    def predict(self, X, k, epsilon=1e-2):
        N = len(X)
        y_hat = np.zeros(N)
        for i in range(N):
            dist = np.sum((self.X - X[i])**2, axis=1) # find distance
            idx = np.argsort(dist)[:k] # get the data indexes whose distances are the closest
            ws = (np.sqrt(dist[idx]) + epsilon)**-1
            y_hat[i] = np.bincount(self.y[idx], weights=ws).argmax()
        return y_hat
