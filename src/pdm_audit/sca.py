"""Native SCA metadata, frozen spectral features, and descriptive correspondence."""
import hashlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.stats import kurtosis

POSITIONS=['DS','FS','Upper','Lower']
MULTIPLES={'IR':'BPFIMultiple','B':'BPFMultiple','OR':'BPFOMultiple'}
ANCHORS=[f'{kind}_{fault}_{h}' for fault in MULTIPLES for h in [1,2] for kind in ['raw','env']]
FEATURES=ANCHORS+['log_RMS','kurtosis','crest_factor']+[f'band_{i}' for i in range(4)]+['RPM','sampling_rate']
EXPECTED={fault:[c for c in ANCHORS if f'_{fault}_' in c] for fault in MULTIPLES}
CLASSES={1:'IR',2:'B',3:'OR'}


def features(signal,fs,rpm,multiples):
    x=np.asarray(signal,dtype=float).ravel()
    if len(x)<128 or not np.isfinite(x).all() or not np.isfinite([fs,rpm]).all() or min(fs,rpm)<=0:
        raise ValueError('invalid_signal_or_acquisition')
    x=x-x.mean();rms=float(np.sqrt(np.mean(x*x)))
    if rms<=0:raise ValueError('zero_signal')
    frequency=np.fft.rfftfreq(len(x),1/fs);window=np.hanning(len(x))
    spectra={'raw':abs(np.fft.rfft(x*window))**2}
    envelope=abs(hilbert(x));envelope-=envelope.mean()
    spectra['env']=abs(np.fft.rfft(envelope*window))**2
    totals={k:float(v[1:].sum()) for k,v in spectra.items()}
    if min(totals.values())<=0:raise ValueError('zero_spectral_denominator')
    out={}
    for fault,multiple in multiples.items():
        for h in [1,2]:
            center=float(multiple)*rpm/60*h
            if center<=0 or center>=fs/2:raise ValueError('unavailable_anchor')
            width=max(.03*center,frequency[1]);selected=(frequency>0)&(abs(frequency-center)<=width)
            if not selected.any():raise ValueError('empty_anchor')
            for kind,power in spectra.items():out[f'{kind}_{fault}_{h}']=float(power[selected].sum()/totals[kind])
    out.update(log_RMS=float(np.log(rms)),kurtosis=float(kurtosis(x,fisher=False)),crest_factor=float(abs(x).max()/rms))
    edges=np.array([0.,.05,.15,.30,.50])*fs
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        selected=(frequency>0)&(frequency>=lo)&(frequency<hi)
        out[f'band_{i}']=float(spectra['raw'][selected].sum()/totals['raw'])
    out.update(RPM=float(rpm),sampling_rate=float(fs))
    if not np.isfinite(list(out.values())).all():raise ValueError('nonfinite_feature')
    return out


def extract(data):
    rows=[];excluded=[];seen={};cross_duplicates=[]
    for case in range(1,12):
        for split in ['train','test']:
            path=data/str(case)/(split+'.mat');mat=loadmat(path,squeeze_me=True,struct_as_record=False)
            for position in POSITIONS:
                if position not in mat:continue
                s=mat[position];raw=np.asarray(s.rawData)
                signals=[np.asarray(x).ravel() for x in raw.ravel()] if raw.dtype==object else list(np.atleast_2d(raw))
                labels=np.atleast_1d(s.label).astype(int);fs=np.atleast_1d(s.samplingRate);rpm=np.atleast_1d(s.RPM);times=np.atleast_1d(s.time)
                assert len(signals)==len(labels)==len(fs)==len(rpm)==len(times)
                multiples={f:float(getattr(s.faultFrequencies,k)) for f,k in MULTIPLES.items()}
                for i,signal in enumerate(signals):
                    name=f'{case}/{split}/{position}/{i}'
                    reason='off_or_missing_speed' if labels[i]==-1 else None
                    if reason is None:
                        try:values=features(signal,float(fs[i]),float(rpm[i]),multiples)
                        except ValueError as e:reason=str(e)
                    if reason is not None:
                        excluded.append({'recording':name,'case':case,'reason':reason});continue
                    assert labels[i] in [0,1,2,3]
                    payload=hashlib.sha256(np.asarray(signal,dtype='<f8').tobytes()).hexdigest()
                    if payload in seen and seen[payload][0]!=case:cross_duplicates.append([seen[payload][1],name])
                    seen.setdefault(payload,(case,name))
                    rows.append({'recording':name,'case':case,'native_split':split,'position':position,'label':int(labels[i]),
                        'measured_at':str(times[i]),'asset':str(mat['assetDescription']),'bearing_model':str(s.assetName),
                        'unit':str(s.unit),'samples':len(signal),'signal_sha256':payload,**{f'multiple_{f}':v for f,v in multiples.items()},**values})
    return pd.DataFrame(rows),excluded,cross_duplicates


def correspondence(attribution,metadata,columns):
    """Equal case/type weights; pooled cross-fitted contrasts include fit variation."""
    table=pd.DataFrame(abs(attribution),columns=columns)
    table['case']=metadata.case.to_numpy();table['label']=metadata.label.to_numpy()
    grouped=table[table.label>0].groupby(['case','label'])[columns].mean()
    result=[]
    for label,fault in CLASSES.items():
        own=grouped.index.get_level_values('label')==label
        count=int(own.sum());rest=~own
        if not own.any():
            result.append({'fault':fault,'available':False,'own_case_types':count,'reason':'missing_own_cohort'});continue
        a=grouped.loc[own].mean();expected=EXPECTED[fault]
        if a.sum()<=0:
            result.append({'fault':fault,'available':False,'own_case_types':count,'reason':'zero_own_attribution'});continue
        a=a/a.sum();relevance=float(a[expected].sum())
        if not rest.any():
            result.append({'fault':fault,'available':False,'own_case_types':count,'R':relevance,'reason':'missing_other_failure_cohort'});continue
        b=grouped.loc[rest].mean()
        if b.sum()<=0:
            result.append({'fault':fault,'available':False,'own_case_types':count,'R':relevance,'reason':'zero_comparison_attribution'});continue
        b=b/b.sum();outside=[x for x in columns if x not in expected]
        excess=(a-b)[outside];feature=str(excess.idxmax())
        result.append({'fault':fault,'available':True,'own_case_types':count,
            'R':relevance,'D':float(a[expected].sum()-b[expected].sum()),
            'S':float(excess[feature]),'S_feature':feature,
            'acquisition_share':float(a[[c for c in ['RPM','sampling_rate'] if c in columns]].sum())})
    return result
