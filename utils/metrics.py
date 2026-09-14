from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
def metrics(y,score,threshold): return {"AUROC":roc_auc_score(y,score),"AUPRC":average_precision_score(y,score),"F1":f1_score(y,score>threshold,zero_division=0)}

