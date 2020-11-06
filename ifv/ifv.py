"""
File: ifv.py
Author: Tim Hilt
Date: 2020-11-03
"""
from random import sample

import numpy as np
from numpy.linalg import norm
from sklearn.mixture import GaussianMixture
import progressbar as pb


class IFV:
    def __init__(self, k=256, n_vocabs=1, alpha=0.5):
        self.k = k
        self.n_vocabs = 1
        self.alpha = alpha
        self.database = None
        self.vocab = None
        self.weights = None
        self.means = None
        self.covariances = None

    def fit(self, X):
        """

        Parameters
        ----------
        X : list(array)
            List of train-descriptor-arrays

        Returns
        -------

        """
        X_mat = np.vstack(X)
        # for i in range(self.n_vocabs):
        gmm = GaussianMixture(n_components=self.k, covariance_type="diag", warm_start=True)

        while not gmm.converged_:
            idx = sample(range(len(X_mat)), int(2e5))
            gmm.fit(X_mat[idx])

        self.weights = gmm.weights_
        self.means = gmm.means_
        self.covariances = gmm.covariances_
        self.vocab = gmm
        self.database = self.transform(X)
        return self

    def predict(self, x):
        """Predict the matching instance for x

        TODO: Find optimal threshold and don't simply accept the argmax!

        Parameters
        ----------
        x : array
            Contains local image descriptors

        Returns
        -------
        argmax(predict_proba(x)) : int
            The index of the predicted instance
        """
        return np.argmax(self.predict_proba(x))

    def predict_proba(self, x):
        """Return probabilities for local image descriptors

        Parameters
        ----------
        x : array
            Contains local image descriptors

        Returns
        -------
        scores : array
            The probabilities x being one instance in self.database
        """
        fisher = self._ifv(x)
        scores = self.database @ fisher
        return scores

    def transform(self, X):
        """

        Parameters
        ----------
        X : list(array)
            List of local image descriptors

        Returns
        -------
        fisher_vectors : array
            each row corresponds to one fisher-vector
        """
        fvs = []
        for i in pb.progressbar(range(len(X))):
            fvs.append(self._ifv(X[i]))
        return np.vstack(fvs)

    def fit_transform(self, X):
        """Subsequentially fit a gmm and transform the

        Parameters
        ----------
        X : list(array)
            List of local descriptor-arrays

        Returns
        -------
        self.transform(X) : array
            The transformed fisher-vectors
        """
        _ = self.fit(X)
        return self.transform(X)

    def _ifv(self, x):
        """

        Parameters
        ----------
        x : array
            Contains local image descriptors

        Returns
        -------

        """
        # Vectorized version: probably buggy
        # s0 = preds.sum(axis=0)
        # s1 = np.einsum("ij,ik->jk", preds, x)
        # s2 = np.einsum("ij,ik->jk", preds, x**2)
        #
        # g_alpha = (s0 - len(x) * self.weights) / np.sqrt(self.weights)
        # g_mu = (s1 - self.means * s0[:, None]) / (np.sqrt(self.weights)[:, None] * self.covariances)
        # g_sigma = (s2 - 2 * self.means * s1 + (self.means**2 - self.covariances**2) * s0[:, None]) / \
        #           (np.sqrt(2 * self.weights[:, None]) * self.covariances**2)
        #
        # fv = np.concatenate((g_alpha, g_mu.flatten(), g_sigma.flatten()))

        # Non-vectorized-version
        T, D = x.shape

        s0 = np.zeros(self.k)
        s1 = np.zeros((self.k, D))
        s2 = np.zeros((self.k, D))

        ytk = self.vocab.predict_proba(x)

        for t in range(T):
            for k in range(self.k):
                s0[k] += ytk[t, k]
                s1[k] += ytk[t, k] * x[t]
                s2[k] += ytk[t, k] * x[t]**2

        G_alpha = np.zeros(self.k)
        G_mu = np.zeros((self.k, D))
        G_sigma = np.zeros((self.k, D))

        for k in range(self.k):
            G_alpha[k] = (s0[k] - T * self.weights[k]) / np.sqrt(self.weights[k])
            G_mu[k] = (s1[k] - self.means[k] * s0[k]) / (np.sqrt(self.weights[k]) * self.covariances[k])
            G_sigma[k] = (s2[k] - 2 * self.means[k] * s1[k] + (self.means[k]**2 - self.covariances[k]**2) * s0[k]) / \
                         (np.sqrt(2 * self.weights[k]) * self.covariances[k])

        fv = np.concatenate((G_alpha, G_mu.flatten(), G_sigma.flatten()))
        fv = self._power_law_norm(fv)
        fv /= norm(fv)
        return fv

    def _power_law_norm(self, X):
        """Perform power-Normalization on a given array

        Parameters
        ----------
        X : array
            Array that should be normalized

        Returns
        -------
        normed : array
            Power-normalized array
        """
        normed = np.sign(X) * np.abs(X)**self.alpha
        return normed

    def _gaussian(self, x, k):
        coeff = (1.0 / ((2.0 * np.pi)**(len(x) / 2) * np.sqrt(np.linalg.det(self.covariances[k]))))
        exponent = np.exp(-0.5 * (x - self.means[k]).T @ np.linalg.pinv(self.covariances[k]) @ (x - self.means[k]))
        return coeff * exponent

    def __repr__(self):
        return f"IFV(k={self.k})"
