import numpy as np
from sklearn.preprocessing import StandardScaler
def load_npz(path,with_labels=False):
    data=np.load(path); result={"train":np.asarray(data["train"],float),"test":np.asarray(data["test"],float)}
    if with_labels: result["test_labels"]=np.asarray(data["test_labels"],int)
    return result
def split_scale(train,ratio=.8):
    cut=int(len(train)*ratio);scaler=StandardScaler().fit(train[:cut]);return scaler.transform(train[:cut]),scaler.transform(train[cut:]),scaler

