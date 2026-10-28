"""Independent source, fold, prediction, attribution and count verification."""
import pdm_audit
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.stats import kurtosis

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';DATA=ROOT/'data/sca'


def measure(y,p,w=None):
    w=np.ones(len(y)) if w is None else np.asarray(w);z=p>=.5
    tp=w[(y==1)&z].sum();fp=w[(y==0)&z].sum();fn=w[(y==1)&~z].sum();tn=w[(y==0)&~z].sum()
    return {'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'healthy_FPR':fp/(fp+tn) if fp+tn else None,
            'recall':tp/(tp+fn) if tp+fn else None,'TP':tp,'FP':fp,'FN':fn,'TN':tn}


def compare_metrics(actual,expected):
    for key,value in expected.items():
        if value is None:assert actual[key] is None
        else:assert np.isclose(actual[key],value,rtol=1e-12,atol=1e-12),key


def check_audit(records,attr,metadata,columns):
    labels=metadata.label.to_numpy();cases=metadata.case.to_numpy()
    groups=[]
    for case in sorted(set(cases)):
        for label in [1,2,3]:
            use=(cases==case)&(labels==label)
            if use.any():groups.append((label,abs(attr[use]).mean(axis=0)))
    for label,fault in [(1,'IR'),(2,'B'),(3,'OR')]:
        saved=next(x for x in records if x['fault']==fault);own=[a for l,a in groups if l==label];rest=[a for l,a in groups if l!=label]
        assert saved['own_case_types']==len(own)
        if not own:assert not saved['available'];continue
        a=np.mean(own,axis=0);a/=a.sum();expected=[i for i,c in enumerate(columns) if f'_{fault}_' in c]
        assert np.isclose(saved['R'],a[expected].sum())
        if not rest:assert not saved['available'] and saved['reason']=='missing_other_failure_cohort';continue
        b=np.mean(rest,axis=0);b/=b.sum();outside=[i for i in range(len(columns)) if i not in expected]
        selected=outside[np.argmax((a-b)[outside])]
        assert saved['available'] and saved['S_feature']==columns[selected]
        assert np.isclose(saved['D'],a[expected].sum()-b[expected].sum()) and np.isclose(saved['S'],(a-b)[selected])
        acquisition=[i for i,c in enumerate(columns) if c in ['RPM','sampling_rate']]
        assert np.isclose(saved['acquisition_share'],a[acquisition].sum())


def main():
    r=json.loads((OUT/'sca_field.json').read_text());assert r['complete']
    for path,value in r['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
    source=json.loads((OUT/'sca_download.json').read_text());metadata=json.loads((DATA/'official_metadata.json').read_text())
    for row in metadata:
        for f in row['files']:
            name=row['folder']['name']+'/'+f['filename'];saved=next(x for x in source['files'] if x['name']==name)
            assert saved['sha256']==f['content_details']['sha256_hash'] and saved['size']==f['content_details']['size']
            with (DATA/name).open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==saved['sha256']
    df=pd.read_csv(OUT/'sca_features.csv');arrays=dict(np.load(OUT/'sca_field_arrays.npz'));lookup={x:i for i,x in enumerate(df.recording)}
    excluded=json.loads((OUT/'sca_feature_gate.json').read_text());excluded_lookup={x['recording']:x['reason'] for x in excluded['excluded']}
    signatures={};cross=[];seen_rows=set();native_count=0
    for file in source['files']:
        case=int(file['name'].split('/')[0]);split=Path(file['name']).stem
        mat=loadmat(DATA/file['name'],squeeze_me=True,struct_as_record=False)
        for position in ['DS','FS','Upper','Lower']:
            if position not in mat:continue
            s=mat[position];raw=np.asarray(s.rawData)
            signals=[np.asarray(x).ravel() for x in raw.ravel()] if raw.dtype==object else list(np.atleast_2d(raw))
            labels=np.atleast_1d(s.label).astype(int);rpm=np.atleast_1d(s.RPM);fs=np.atleast_1d(s.samplingRate);times=np.atleast_1d(s.time)
            feature_checked=False
            for j,signal in enumerate(signals):
                name=f'{case}/{split}/{position}/{j}';native_count+=1
                if name in excluded_lookup:
                    assert labels[j]==-1 and excluded_lookup[name]=='off_or_missing_speed';continue
                i=lookup[name];seen_rows.add(i);record=df.iloc[i]
                assert record.label==labels[j] and record.case==case and record.native_split==split
                assert record.samples==len(signal)
                assert record.measured_at==str(times[j]) and record.RPM==rpm[j] and record.sampling_rate==fs[j]
                payload=hashlib.sha256(np.asarray(signal,dtype='<f8').tobytes()).hexdigest();assert payload==record.signal_sha256
                if payload in signatures and signatures[payload][0]!=case:cross.append([signatures[payload][1],name])
                signatures.setdefault(payload,(case,name))
                if not feature_checked:
                    # Independent FFT-band arithmetic on one native row per position/file.
                    x=np.asarray(signal,dtype=float);x=x-x.mean();window=np.hanning(len(x));frequency=np.fft.rfftfreq(len(x),1/fs[j])
                    envelope=abs(hilbert(x));envelope-=envelope.mean()
                    spectra={'raw':abs(np.fft.rfft(x*window))**2,'env':abs(np.fft.rfft(envelope*window))**2}
                    for fault,key in [('IR','BPFIMultiple'),('B','BPFMultiple'),('OR','BPFOMultiple')]:
                        multiple=float(getattr(s.faultFrequencies,key));assert np.isclose(record['multiple_'+fault],multiple)
                        for h in [1,2]:
                            center=multiple*rpm[j]/60*h;width=max(.03*center,frequency[1]);mask=(frequency>0)&(abs(frequency-center)<=width)
                            for kind,power in spectra.items():assert np.isclose(record[f'{kind}_{fault}_{h}'],power[mask].sum()/power[1:].sum(),rtol=1e-10,atol=1e-12)
                    rms=np.sqrt(np.mean(x*x));assert np.isclose(record.log_RMS,np.log(rms))
                    assert np.isclose(record['kurtosis'],kurtosis(x,fisher=False)) and np.isclose(record.crest_factor,abs(x).max()/rms)
                    for k,(lo,hi) in enumerate(zip([0,.05,.15,.3],[.05,.15,.3,.5])):
                        mask=(frequency>0)&(frequency>=lo*fs[j])&(frequency<hi*fs[j]);power=spectra['raw']
                        assert np.isclose(record[f'band_{k}'],power[mask].sum()/power[1:].sum())
                    feature_checked=True
    assert len(seen_rows)==len(df)==r['records'] and not cross and native_count==len(df)+len(excluded_lookup)
    y=(df.label.to_numpy()>0).astype(int);cases=df.case.to_numpy();external=np.flatnonzero(cases==11);use=np.flatnonzero(cases<=10)
    for item in r['summary']:
        view=item['view'];columns=item['columns'];p=arrays[view+'_probabilities'];attr=arrays[view+'_shap'];offset=arrays[view+'_expected']
        proof=next(x for x in r['proof'] if x['view']==view)
        assert np.array_equal(np.flatnonzero(np.isfinite(p)),use)
        for j,fold in enumerate(proof['folds']):
            held=fold['held_case'];train=np.asarray(fold['train_rows']);test=np.asarray(fold['test_rows']);background=np.asarray(fold['background_rows'])
            assert np.array_equal(train,np.flatnonzero((cases<=10)&(cases!=held))) and np.array_equal(test,np.flatnonzero(cases==held))
            assert len(background)==96 and set(background)<=set(train) and not set(background)&set(test)
            expected_background=[];training_cases=sorted(set(cases[train]))
            for k,c in enumerate(training_cases):
                rows=train[cases[train]==c];count=96//len(training_cases)+(k<96%len(training_cases))
                expected_background.extend(rows[np.linspace(0,len(rows)-1,count,dtype=int)])
            assert np.array_equal(background,expected_background)
            path=ROOT/fold['model'];assert hashlib.sha256(path.read_bytes()).hexdigest()==fold['model_sha256'];model=joblib.load(path)
            assert model.n_estimators==64 and model.max_depth==6 and model.min_samples_leaf==5 and model.random_state==2026
            assert np.allclose(model.predict_proba(df.iloc[test][columns])[:,1],p[test])
            assert np.allclose(model.predict_proba(df.iloc[external][columns])[:,1],arrays[view+'_external_probabilities'][j])
            assert np.max(abs(attr[test].sum(axis=1)+offset[test]-p[test]))<1e-6
            compare_metrics(item['fold_metrics'][j],measure(y[test],p[test]))
            check_audit(item['within_case_audit'][j]['audit'],attr[test],df.iloc[test],columns)
        weights=np.array([1/np.count_nonzero(cases[use]==c) for c in cases[use]])
        compare_metrics(item['case_weighted_metrics'],measure(y[use],p[use],weights))
        check_audit(item['audit'],attr[use],df.iloc[use],columns)
        mean=arrays[view+'_external_probabilities'].mean(axis=0)
        for record in item['external']:
            selected=df.iloc[external].native_split.to_numpy()==record['native_split']
            assert selected.sum()==record['records'] and not y[external[selected]].any()
            compare_metrics(record,measure(y[external[selected]],mean[selected]))
    assert not any(r[k] for k in ['independent_sessions_verified','exhaustive_physical_alignment_verified','calibrated_industrial_warning'])
    paths=[OUT/'sca_field.json',OUT/'sca_field_arrays.npz',OUT/'sca_features.csv',Path(__file__)]
    manifest={'verified':True,'native_measurements':native_count,'retained':len(df),'files':len(source['files']),
        'cases':11,'independent_sessions_verified':False,'physical_alignment_verified':False,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'sca_field_verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS publisher hashes, native field labels/features, 20 case-held-out models, training backgrounds, SHAP additivity, all metrics and cohort abstentions')


if __name__=='__main__':main()
