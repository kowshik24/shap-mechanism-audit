"""Whole-trajectory correspondence scores and continuous response comparator."""
import numpy as np


def score_packet(contributions,mechanism,expected,context,normalize):
    a=np.asarray(contributions);statistics=[];valid=True
    for fault,indices in expected.items():
        own=mechanism==fault;rest=(mechanism>0)&~own
        if not own.any() or not rest.any():valid=False;continue
        left=a[own].mean(axis=0);right=a[rest].mean(axis=0)
        if normalize:
            if min(left.sum(),right.sum())<=0:valid=False;continue
            left=left/left.sum();right=right/right.sum()
        outside=[j for j in range(a.shape[1]) if j not in indices]
        index=outside[int(np.argmax((left-right)[outside]))]
        item={'mechanism':fault,'S':float((left-right)[index]),'S_feature_index':index}
        if normalize:item.update(R=float(left[indices].sum()),D=float((left-right)[indices].sum()))
        statistics.append(item)
    maximum=max((x['S'] for x in statistics),default=np.inf)
    local=max((x['S'] for x in statistics if x['S_feature_index']==context),default=-np.inf)
    return {'score':maximum if valid else np.inf,'localization':local if valid else -np.inf,
            'valid':valid,'statistics':statistics}


def prediction_response(model,x,median):
    n,d=x.shape;queries=np.tile(x,(d,1))
    for j in range(d):queries[j*n:(j+1)*n,j]=median[j]
    baseline=model.predict_proba(x)[:,1]
    changed=model.predict_proba(queries)[:,1].reshape(d,n).T
    return abs(baseline[:,None]-changed)


def predictive_metrics(y,probability):
    z=np.asarray(probability)>=.5;y=np.asarray(y,bool)
    tp=int((z&y).sum());fp=int((z&~y).sum());fn=int((~z&y).sum());tn=int((~z&~y).sum())
    recall=tp/(tp+fn);specificity=tn/(tn+fp)
    return {'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'healthy_FPR':1-specificity,
            'balanced_accuracy':(recall+specificity)/2,'TP':tp,'FP':fp,'FN':fn,'TN':tn}
