"""Prespecified, recording-weighted KAIST load-transfer helpers."""
import numpy as np
from scipy.signal import periodogram
from scipy.stats import kurtosis, skew
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

SEED = 20261026
CLASSES = ['Normal', 'BPFI', 'BPFO']
BANDS = [(0,500),(500,1000),(1000,2000),(2000,4000),(4000,None)]


def features(window, fs):
    """Forty fixed generic features; timestamp and native load excluded."""
    result = {}
    for j in range(4):
        x = np.asarray(window[:,j], dtype=float)
        x = x-x.mean()
        rms = np.sqrt(np.mean(x*x))
        if rms <= 0 or not np.isfinite(x).all():
            raise ValueError('Nonfinite or constant vibration window')
        prefix = f'ch{j+1}_'
        result.update({prefix+'log_rms': float(np.log(rms)),
                       prefix+'log_std': float(np.log(x.std())),
                       prefix+'kurtosis': float(kurtosis(x, fisher=False)),
                       prefix+'skewness': float(skew(x)),
                       prefix+'crest': float(np.max(abs(x))/rms)})
        frequency, power = periodogram(x, fs=fs, window='hann')
        for lo, hi in BANDS:
            mask = (frequency >= lo) & ((frequency < hi) if hi else np.ones(len(frequency),bool))
            result[prefix+f'band_{lo}_{hi or "nyquist"}'] = float(power[mask].sum()/power.sum())
    assert len(result) == 40 and np.isfinite(list(result.values())).all()
    return result


def weights(frame):
    """Equal classes, equal native records within class; normalize to mean one."""
    sizes = frame.groupby('recording').size()
    counts = frame.groupby('label')['recording'].nunique()
    values = np.array([1/(sizes[r]*counts[y]) for r,y in zip(frame.recording,frame.label)])
    return values/values.mean()


def metrics(frame, predicted):
    y = frame.label.to_numpy()
    w = weights(frame)
    predicted = np.asarray(predicted)
    binary, binary_prediction = y != 0, predicted != 0
    return dict(macro_F1=float(f1_score(y,predicted,labels=[0,1,2],average='macro',sample_weight=w)),
                balanced_accuracy=float(balanced_accuracy_score(y,predicted,sample_weight=w)),
                binary_balanced_accuracy=float(balanced_accuracy_score(binary,binary_prediction,sample_weight=w)),
                healthy_FPR=float(np.average(binary_prediction[y==0],weights=w[y==0])),
                per_class_recall=np.diag(confusion_matrix(y,predicted,labels=[0,1,2],sample_weight=w,normalize='true')).tolist(),
                confusion=confusion_matrix(y,predicted,labels=[0,1,2],sample_weight=w,normalize='true').tolist())


def fit_predict(train, test, names, family, parameter):
    w = weights(train)
    scaler = StandardScaler().fit(train[names], sample_weight=w)
    if family == 'logistic':
        model = LogisticRegression(C=parameter, max_iter=2000, random_state=SEED)
    else:
        model = RandomForestClassifier(n_estimators=200,max_depth=parameter,
            min_samples_leaf=2,max_features='sqrt',random_state=SEED,n_jobs=2)
    model.fit(scaler.transform(train[names]),train.label,sample_weight=w)
    return model.predict(scaler.transform(test[names])), model, scaler
