import numpy as np

class Binomial_Classifier():
    ''' Description
    This is a supervised learning algorithm, Binomial Naive Bayes.
    In fit function:
        The model learns the  knowledge(the FREQUENCIES OF 1 and 0)
            from the dataset, and uses the knowledge to predict new data.
        EACH data sample contains:
            an input and a label, where the input is
            a set of features(1s or 0s), such as [x1,x2,...xi...],
            let's say there are 1000 features for each sample.
        For EACH category:
            the model learns A and Prior
            where A is (the frequencies of 1)/(the frequencies of 0).
            Then A = (A+epsilon)/sum(A+epsilon), where the numerator
            is a vector with size 1000, and denominator is a number.
            A has the same size as a sample, 1000.
            The Prior = (# of samples in this category)/(entire dataset)
        Thus, K categories, K As and K Priors.
        The model uses As and Priors to predict new samples.
    In predict funcion:
        posterior = p(1-p)prior
        Let's say we have 800 new samples.
        For EACH category:
            the model generates 800 probabilities/scores.
        Thus, we get 800 by K probabilities, where K is
            the number of the categories.
        For each row:
            we find the index of the max probability,
            so the category we want to assign to the sample
            is that index.
    '''
    def fit(self, X, y, epsilon=1):
        self.priors = dict()
        self.likelihoods = dict()
        self.K = set(y.astype(int))
        for k in self.K:
            X_k = X[y == k]
            self.priors[k] = len(X_k)/len(X)
            self.likelihoods[k] = (np.count_nonzero(X_k, 0)+epsilon)/sum(np.count_nonzero(X_k, 0)+epsilon)

    def predict(self, X):
        N, D = X.shape
        P_hat = np.zeros((N, len(self.K)))
        for k, l in self.likelihoods.items():
            P_hat[:,k] = np.sum(X*np.log(l),1)+np.sum((1-X)*np.log(1-l),1)+np.log(self.priors[k])
        return np.argmax(P_hat, 1).astype(int)

    def get_confusion(self, X, y):
        n = len(self.K)
        conf_matrix = np.zeros((n,n))
        for i in range(n):
            y_hat = self.predict(X[y==i])
            for j in range(n):
                conf_matrix[i,j] = sum(y_hat==j)
            conf_matrix[i] = conf_matrix[i]/sum(conf_matrix[i])
        return np.round(conf_matrix, 2)
