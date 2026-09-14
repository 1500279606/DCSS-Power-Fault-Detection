import numpy as np
from sklearn.covariance import LedoitWolf

class InnovationScore:
    def __init__(self,regularization=1e-4): self.regularization=regularization
    def fit(self,posterior,prior):
        e=np.asarray(posterior)-np.asarray(prior);e=e[np.isfinite(e).all(1)];self.mean=e.mean(0)
        cov=LedoitWolf().fit(e).covariance_+self.regularization*np.eye(e.shape[1]);self.precision=np.linalg.pinv(cov);return self
    def score(self,posterior,prior):
        e=np.asarray(posterior)-np.asarray(prior);d=e-self.mean
        return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n",d,self.precision,d),0))

class RobustNormalizer:
    def fit(self,x):
        x=np.asarray(x);x=x[np.isfinite(x)];self.median=np.median(x);self.scale=max(1.4826*np.median(abs(x-self.median)),1e-4);return self
    def transform(self,x): return (np.asarray(x)-self.median)/self.scale

