"""Regime-specific lagged directed predictive-dependency estimation."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge

def lagged_design(x, lag):
    phi=np.concatenate([x[lag-l:-l] for l in range(1,lag+1)],axis=1)
    return phi,x[lag:]

def mechanism(x, lag, alpha):
    phi,y=lagged_design(x,lag); full=Ridge(alpha=alpha).fit(phi,y)
    base=((y-full.predict(phi))**2).mean(axis=0); d=x.shape[1]
    strength=np.zeros((lag,d,d))
    for l in range(lag):
        for source in range(d):
            keep=np.ones(lag*d,bool);keep[l*d+source]=False
            reduced=Ridge(alpha=alpha).fit(phi[:,keep],y)
            strength[l,source]=np.maximum(((y-reduced.predict(phi[:,keep]))**2).mean(axis=0)-base,0)
    return strength

class DynamicDependency:
    def __init__(self, regimes=3, max_lag=2, top_k=3, ridge_alpha=1., window=256, stride=64):
        self.regimes,self.max_lag,self.top_k,self.alpha=regimes,max_lag,top_k,ridge_alpha
        self.window,self.stride=window,stride
    def fit(self,x):
        starts=list(range(0,len(x)-self.window+1,self.stride)); starts+=[len(x)-self.window]
        starts=sorted(set(starts)); features=np.stack([mechanism(x[s:s+self.window],self.max_lag,self.alpha).ravel() for s in starts])
        labels=AgglomerativeClustering(n_clusters=min(self.regimes,len(features)),linkage="average",metric="euclidean").fit_predict(features)
        self.prototypes=np.stack([features[labels==k].mean(0) for k in range(labels.max()+1)])
        self.graphs=[]
        for k in range(len(self.prototypes)):
            strength=self.prototypes[k].reshape(self.max_lag,x.shape[1],x.shape[1]); mask=np.zeros_like(strength)
            for target in range(x.shape[1]):
                flat=strength[:,:,target].ravel(); idx=np.argsort(flat)[-self.top_k:]
                for position in idx:
                    lag,source=np.unravel_index(position,(self.max_lag,x.shape[1]))
                    mask[lag,source,target]=float(flat[position]>0)
            self.graphs.append(strength*mask)
        self.graphs=np.stack(self.graphs); return self
    def assign(self,x):
        result=np.zeros(len(x),dtype=int)
        starts=list(range(0,max(1,len(x)-self.window+1),self.stride))
        starts.append(max(0,len(x)-self.window))
        for start in sorted(set(starts)):
            stop=min(start+self.window,len(x)); block=x[max(0,stop-self.window):stop]
            if len(block)<=self.max_lag: continue
            h=mechanism(block,self.max_lag,self.alpha).ravel(); regime=np.linalg.norm(self.prototypes-h,axis=1).argmin()
            result[start:stop]=regime
        return result
    def messages(self,x,regimes):
        out=np.zeros((len(x)-self.max_lag,self.max_lag*x.shape[1]))
        for t in range(self.max_lag,len(x)):
            out[t-self.max_lag]=np.concatenate([x[t-l]@self.graphs[regimes[t],l-1] for l in range(1,self.max_lag+1)])
        return out
