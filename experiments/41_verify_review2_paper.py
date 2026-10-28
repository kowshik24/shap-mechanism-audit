"""Verify the current reference-study manuscript and native LaTeX result assets.

The filename is retained for older runner compatibility. Scientific arithmetic
is independently checked by experiment 45 before these manuscript bindings.
"""
import pdm_audit
import hashlib,json,re,subprocess
from pathlib import Path
import numpy as np
from pdm_audit.conditional_audit import observed_overlap
from pdm_audit.reference_transfer import scores
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'IEEE-conference-template-062824';OUT=ROOT/'experiments/results'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def coordinates(source):
    return [[tuple(map(float,p)) for p in re.findall(r'\(([^,]+),([^\)]+)\)',block)] for block in re.findall(r'coordinates \{([^}]+)\}',source)]


def main():
    tex=PAPER/'IEEE-conference-template-062824.tex';pdf=tex.with_suffix('.pdf');source=tex.read_text()
    assert re.findall(r'\\section\{([^}]+)\}',source)==['Introduction','Related Work and Background','Methodology','Results','Experimental Analysis','Conclusion','Future Work']
    title=re.search(r'\\title\{([^}]+)\}',source).group(1);metadata=json.loads((PAPER/'submission_metadata.json').read_text())
    assert title==metadata['title'] and not metadata['portal_verified']
    assert metadata['abstract_latex']==re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',source,re.S).group(1).strip()
    proof=json.loads((OUT/'reference_transfer_verification.json').read_text());assert proof['verified']
    for p,value in proof['sha256'].items():assert digest(ROOT/p)==value,p
    matched_proof=json.loads((OUT/'matched_population_verification.json').read_text());assert matched_proof['verified']
    for p,value in matched_proof['sha256'].items():assert digest(ROOT/p)==value,p
    field_proof=json.loads((OUT/'sca_field_verification.json').read_text());assert field_proof['verified']
    for p,value in field_proof['sha256'].items():assert digest(ROOT/p)==value,p
    for name in ['dtu_prospective_verification.json','tree_precision_verification.json','chamber_verification.json','partial_map_verification.json']:
        extra=json.loads((OUT/name).read_text());assert extra['verified']
        for p,value in extra['sha256'].items():assert digest(ROOT/p)==value,p
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));assert pages<=10
    assert re.search(r'^Title:\s+(.+)$',info,re.M).group(1)==title and pdf.stat().st_mtime>=tex.stat().st_mtime
    log=tex.with_suffix('.log').read_text();assert not any(x in log for x in ['Overfull','undefined','multiply defined'])
    text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True);assert '??' not in text
    assert not re.search(r'\b(?:TODO|PENDING|PLACEHOLDER)\b',text) and '\\appendix' not in source
    assert not re.search(r'(?<!\\)%',source)
    bibliography=(PAPER/'refs.bib').read_text();keys=re.findall(r'@\w+\s*\{\s*([^,]+)',bibliography);assert len(keys)==len(set(keys))
    for group in re.findall(r'\\cite\{([^}]+)\}',source):assert set(group.split(','))<=set(keys)
    for token in ['14.1','0.121','0.390','0.0625','0.313','6.89','7.72']:assert token in text,token
    assert not any(x in source for x in ['review-driven','cluster-bootstrap amendment','in this revision','reproduce\\_review2.sh'])
    r=json.loads((OUT/'reference_transfer.json').read_text());c=json.loads((OUT/'composite_reference.json').read_text());w=json.loads((OUT/'cwru_attribution_audit.json').read_text());d=np.load(OUT/'reference_transfer_draws.npz');f1=d['f1'];targeted=d['targeted']
    tables={name:(PAPER/'tables'/name).read_text() for name in ['transfer_warning.tex','transfer_harm.tex','transfer_bins.tex','cwru_descriptive_audit.tex','matched_population.tex','sca_field.tex','dtu_prospective.tex']}
    prospective=json.loads((OUT/'dtu_prospective.json').read_text())
    for name in ['dtu_prospective.json','dtu_prospective_source.json','tree_precision_audit.json']:
        saved=json.loads((OUT/name).read_text())
        for path,value in saved['sha256'].items():assert digest(ROOT/path)==value,path
    for record in prospective['records']:
        assert digest(ROOT/record['model'])==record['model_sha256']
    for record in json.loads((OUT/'dtu_prospective_source.json').read_text())['records']:
        assert digest(ROOT/record['raw_path'])==record['raw_sha256']
    physical=json.loads((OUT/'chamber_physical_map.json').read_text())
    for name in ['chamber_sources.json','chamber_physical_map.json','chamber_assets.json']:
        record=json.loads((OUT/name).read_text())
        for path,value in record['sha256'].items():assert digest(ROOT/path)==value,path
    acquisitions=json.loads((OUT/'chamber_acquisitions.json').read_text())
    assert len(acquisitions)==57 and sum(x['passed'] for x in acquisitions)==32
    assert len(physical['pairs'])==627 and physical['all_pairs_assessed']
    assert '57 intervention records; 32 replay' in source and '627 intervention/channel pairs' in source
    assert all(not p['causally_excluded'] for p in physical['pairs'])
    assert 'Missing graph paths and non-rejections remain unresolved' in source
    partial=json.loads((OUT/'partial_map.json').read_text())
    for name in ['partial_map.json','partial_map_assets.json','publisher_protocol_replay.json','publisher_runtime_check.json']:
        record=json.loads((OUT/name).read_text())
        for path,value in record['sha256'].items():assert digest(ROOT/path)==value,path
    assert len(partial['rows'])==715 and '\\subsection{Incomplete Feature Maps}' in source
    assert '\\label{eq:map-ranges}' in source and 'not sampling intervals' in source
    first=next(x for x in partial['rows'] if x['study']=='CWRU' and x['identifier']=='original/matched/B')
    for key in ['R','D','S']:
        for endpoint in first['bounds'][key]:
            if isinstance(endpoint,str):assert '-\\infty' in source
            elif not np.isclose(endpoint,1):assert f'{endpoint:.3f}' in source
    assert all(x['causal_warning_abstentions']==60 for x in partial['rank_summary'])
    for view,label in [('unaltered','Unaltered'),('matched','Biased, matched'),('removed','Bias removed'),('reversed','Bias reversed')]:
        a,b=[next(x for x in prospective['summary'] if x['score']==kind and x['view']==view) for kind in ['shap','response']]
        m=a['mean_metrics'];row=[label,f"{m['F1']:.3f}",f"{m['balanced_accuracy']:.3f}",f"{a['warning_count']}/{a['localization_count']}",f"{b['warning_count']}/{b['localization_count']}"]
        assert ' & '.join(row) in tables['dtu_prospective.tex']
    assert sum(len(r['results'])*128 for r in prospective['records'])==43520 and '43,520' in source
    assert '0--5.96' in source and '8.92' in source and '0.857' in source
    duration=(PAPER/'tables/dtu_runtime.tex').read_text()
    extraction=json.loads((OUT/'dtu_prospective_source.json').read_text())
    assert '{'+f"{prospective['seconds']:.1f}"+'}' in duration
    assert '{'+f"{extraction['seconds']:.0f}"+'}' in duration
    field=json.loads((OUT/'sca_field.json').read_text())
    for x in field['summary']:
        metrics=x['case_weighted_metrics'];external=next(v for v in x['external'] if v['native_split']=='test')
        row=['12 anchors' if x['view']=='anchors' else '21 features']+[f'{metrics[k]:.3f}' for k in ['F1','healthy_FPR','recall']]+[f"{external['healthy_FPR']:.3f}"]
        assert ' & '.join(row) in tables['sca_field.tex']
        assert all(not a['available'] for z in x['within_case_audit'] for a in z['audit'])
    assert str(field['records'])=='6446' and '6,446' in source and '198' in source
    for a in next(x for x in field['summary'] if x['view']=='all')['audit']:
        assert f"{a['R']:.3f}" in source
    assert 'not an audit of one fixed predictor' in source and 'healthy prediction false-positive rates' in source
    assert '0.300/0.299' in source and '13.4/12.4' in source
    matched=json.loads((OUT/'matched_population.json').read_text())
    for x in matched['summary']:
        if x['n']!=10000:continue
        a,b=x['scores'];delta=x['flip_minus_shap'][1][2];lo,hi=x['difference_ci95'][1][2]
        row=[x['family'].title(),'Global' if x['method']=='global' else 'Fixed bins']
        row += [f"{100*s['matched_localization'][law][2]:.1f}" for law in [0,1] for s in [a,b]]
        row += [f'{100*delta:.1f} [{100*lo:.1f}, {100*hi:.1f}]']
        assert ' & '.join(row) in tables['matched_population.tex']
    assert '400 paired bootstrap' in source and 'retrospective' in source
    def find(n,f,k,m):return next(x for x in r['summary'] if (x['n'],x['family'],x['score'],x.get('method'))==(n,f,k,m))
    for n in [1000,10000]:
        for f in ['tree','forest','logistic']:
            a=find(n,f,'shap','fixed3');b=find(n,f,'flip','fixed3');co=next(x for x in c['summary'] if (x['n'],x['family'],x['score'],x['method'])==(n,f,'shap','fixed3'))
            cross=[x['mean']*100 for x in a['aligned_warning'][1:]]
            row=[f'{n:,}',f.title(),f"{100*a['self_warning']['mean']:.2f}",f"{100*a['aligned_warning'][0]['mean']:.2f}",f'{min(cross):.2f}--{max(cross):.2f}',f"{100*co['null_warning']['mean']:.2f}"]
            assert ' & '.join(row) in tables['transfer_warning.tex']
            row=[f'{n:,}',f.title(),str(a['harm_cases']),f"{a['laws'][3]['mean_F1_loss']:.3f}"]
            row += [f"{100*x['laws'][law]['harm_localization']['mean']:.1f}" for x in [a,b] for law in [0,3]]
            assert ' & '.join(row) in tables['transfer_harm.tex']
    expected=[]
    for row in range(2):
        for f in ['tree','forest','logistic']:
            a=find(10000,f,'shap','fixed3');b=find(10000,f,'flip','fixed3');ids=[i for i,x in enumerate(r['records']) if x['n']==10000 and x['family']==f]
            harm=f1[ids,0].mean(axis=1)-f1[ids,3].mean(axis=1)>=.05
            values=([v['harm_localization']['mean'] for v in a['laws']],[v['harm_localization']['mean'] for v in b['laws']],targeted[ids].mean(axis=2)[harm].mean(axis=0)) if row==0 else ([v['mean_F1_loss'] for v in a['laws']],)
            expected.extend([list(zip(r['laws'],v)) for v in values])
    found=coordinates((PAPER/'figures/transfer_harm.tex').read_text());assert len(found)==len(expected)
    for a,b in zip(found,expected):assert np.allclose(a,b,atol=1e-14)
    original=np.load(OUT/'observed_overlap_draws.npz');counts=original['target_counts'];found=coordinates((PAPER/'figures/overlap_score.tex').read_text())
    for panel,f in zip(found,['tree','forest','logistic']):
        row=next(x for x in r['records'] if x['n']==10000 and x['family']==f);bank=counts[row['array_index']];o=observed_overlap(bank);a=scores(row['reference_table'],bank)[0];expected=[]
        for lo,hi in zip(np.linspace(.8,1,11)[:-1],np.linspace(.8,1,11)[1:]):
            use=(o>=lo)&((o<hi)|((hi==1)&(o<=hi)))
            if use.any():expected.append((o[use].mean(),a[use].mean()))
        assert np.allclose(panel,expected,atol=1e-14)
    for f in ['tree','forest','logistic']:
        entries=[find(10000,f,'shap',m) for m in r['methods']]
        row=[f.title(),f"{100*entries[1]['eta2_mean']:.1f}"]+[f"{100*x['laws'][0]['all_localization']['mean']:.1f}" for x in entries]
        assert ' & '.join(row) in tables['transfer_bins.tex']
    for x in w['records']:
        setting='Original' if x['condition']=='original' else ('Resonance' if x['view']=='matched' else 'Removed')
        feature=x['S_feature']
        if feature.startswith('band_'):
            lo,hi=map(int,feature.split('_')[1:]);name=f'{lo//1000}--{min(hi,6000)//1000} kHz band'
        elif feature.startswith('raw_'):name=feature.split('_')[1]+' raw, first harmonic'
        else:name={'log_RMS':'log RMS','kurtosis':'Kurtosis'}[feature]
        row=[setting,x['fault']]+[f"{x[k]:.3f}" for k in ['R','D','S']]+[name,f"{x['band_share']:.3f}"]
        assert ' & '.join(row) in tables['cwru_descriptive_audit.tex']
    runtime=(PAPER/'tables/reference_runtime.tex').read_text()
    durations=[x['tree_shap_seconds'] for p in w['proof'] if 'folds' in p for x in p['folds']]
    assert f'{min(durations):.2f}--{max(durations):.2f}' in runtime and w['environment']['cpu'] in runtime
    assert '\\includegraphics' not in source
    assets=[PAPER/'tables/reference_runtime.tex',PAPER/'tables/dtu_runtime.tex']+[PAPER/'figures'/x for x in ['audit_workflow.tex','transfer_harm.tex','overlap_score.tex']]+[PAPER/'tables'/x for x in tables]
    for path in assets:
        assert path.is_file()
        if path.name!='sca_field.tex':assert str(path.relative_to(PAPER)) in source
    assert 'tables/sca_field.tex' not in source
    files=[tex,pdf,PAPER/'refs.bib',PAPER/'submission_metadata.json',Path(__file__),ROOT/'experiments/46_reference_transfer_assets.py',ROOT/'experiments/50_matched_population_assets.py',ROOT/'experiments/57_sca_assets.py',ROOT/'experiments/64_dtu_assets.py',OUT/'reference_transfer_verification.json',OUT/'matched_population_verification.json',OUT/'sca_field_verification.json',OUT/'dtu_prospective_verification.json',OUT/'tree_precision_verification.json',OUT/'chamber_verification.json',OUT/'chamber_assets.json',OUT/'partial_map_verification.json',OUT/'partial_map_assets.json']+assets
    manifest={'verified':True,'pdf_pages':pages,'title':title,'portal_verified':False,'scope':'Current reference-transfer and harm study',
              'sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}}
    (OUT/'review3_paper_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'PASS current reference-study paper: {pages} pages, three native figures, six result tables, partial-map bounds, metadata/citations and verified scientific inputs')


if __name__=='__main__':main()
