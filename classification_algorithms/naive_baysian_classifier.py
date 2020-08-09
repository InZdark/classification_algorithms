import numpy as np
from scipy.stats import multivariate_normal as mvn

class NB_Classifier():
    ''' Description
    This is a supervised learning algorithm, Gaussian Naive Bayes.
    In fit function:
        The model learns the  knowledge(MEANS and VARIANCES)
            from the dataset, and uses the knowledge to predict new data.
        EACH data sample contains:
            an input and a label, where the input is
            a set of features, such as [x1,x2,...xi...],
            let's say there are 1000 features for each sample.
        For EACH category:
            the model learns A, B, and Prior
            where A is the mean features which has the
            same length as a sample, 1000.
            where B is the variance features which also
            has the same length as a sample, 1000.
            The Prior = (# of samples in this category)/(entire dataset)
        Thus, K categories, K As, K Bs, and K Priors.
        The model uses As, Bs, and Priors to predict new samples.
    In predict funcion:
        mvn stands for multivariate normal/Gaussian.
        Let's say we have 800 new samples.
        For EACH category:
            the mvn generates 800 probabilities/scores.
        Thus, we get 800 by K probabilities, where K is
            the number of the categories.
        For each row:
            we find the index of the max probability,
            so the category we want to assign to the sample
            is that index.
    '''
    def fit(self, X, y, epsilon=1e-3):
        self.likelihoods = dict()
        self.priors = dict()
        self.K = set(y.astype(int))
        for k in self.K:
            X_k = X[y==k,:]
            self.likelihoods[k] = {'means': X_k.mean(axis=0),\
                                   'cov': X_k.var(axis=0)+epsilon}
            self.priors[k] = len(X_k)/len(X)

    def predict(self, X, return_prob=False):
        N, D = X.shape
        P_hat = np.zeros((N, len(self.K)))
        for k, l in self.likelihoods.items():
            P_hat[:,k] = mvn.logpdf(X, l['means'], l['cov']) + np.log(self.priors[k])
        if return_prob:
            P_hat
        else:
            return np.argmax(P_hat, axis=1).astype(int)

    def get_confusion(self, X, y):
        n = len(self.K)
        confusion_matrix = np.zeros((n,n))
        for k in range(n):
            y_hat = self.predict(X[y==k])
            for j in range(n):
                confusion_matrix[k,j] = sum(y_hat==j)
            confusion_matrix[k] = confusion_matrix[k]/sum(confusion_matrix[k])
        return np.round(confusion_matrix,2)
