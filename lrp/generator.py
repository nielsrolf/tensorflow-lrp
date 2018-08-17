from lrp import *
import pdb


class GaussianGenerator():
    """
    Generaotr for gaussian data, estimates mean and cov matrix
    """
    def __init__(self, X):
        self.sample_shape = X[0].shape
        X = np.reshape(X, [X.shape[0], np.prod(X[0].shape)])
        self.mean = np.mean(X, axis=0)
        self.cov = np.cov(X.T) + 0.1*np.eye(X.shape[1])
        print("cov", self.cov.shape)

    def __call__(self, X, features):
        """
        1: known, 0: unkown
        m: mean
        c: cov
        k: known
        u: unknown
        u_k: unkown given known
        """
        f = np.reshape(features, np.prod(features.shape))
        k, u = np.where(f==1)[0], np.where(f==0)[0]
        m_k, m_u = self.mean[k], self.mean[u]
        cov_kk, cov_uu = self.cov[k][:,k], self.cov[u][:,u]
        cov_uk = self.cov[u][:,k]
        X_k = X[:,k].T
        # X_k holds different samples in different columns
        m_u_k = m_u[..., None] + cov_uk.dot(np.linalg.inv(cov_kk)).dot(X_k - m_k[...,None])
        # m_u_k also holds the different mean vectors in the different columns
        cov_u_k = cov_uu - cov_uk.dot(np.linalg.inv(cov_kk)).dot(cov_uk.T)
        gen = np.array(X)
        for i in range(X.shape[0]):
            gen[i][u] = np.random.multivariate_normal(m_u_k[:,i], cov_u_k)
        return np.reshape(gen, [gen.shape[0]] + list(self.sample_shape))


class StupidGenerator():
    def __init__(self, X):
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

    def __call__(self, X, features):
        X = np.array(X)
        f = np.reshape(features, np.prod(features.shape))
        k, u = np.where(f==1)[0], np.where(f==0)[0]
        X[:,u] = np.random.normal(self.mean[u], self.var[u])
        return X



