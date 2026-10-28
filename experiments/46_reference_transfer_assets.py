"""Export the reference-transfer evidence as native LaTeX tables and vector plots."""
import pdm_audit
import json
from pathlib import Path
import numpy as np
from pdm_audit.reference_transfer import scores
from pdm_audit.conditional_audit import observed_overlap
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';PAPER=ROOT/'IEEE-conference-template-062824'


def table(name,columns,header,rows):
    lines=[r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines +=[r'\bottomrule',r'\end{tabular}'];(PAPER/'tables'/name).write_text('\n'.join(lines)+'\n')


def main():
    r=json.loads((OUT/'reference_transfer.json').read_text());c=json.loads((OUT/'composite_reference.json').read_text());w=json.loads((OUT/'cwru_attribution_audit.json').read_text())
    raw=np.load(OUT/'reference_transfer_draws.npz');f1=raw['f1'];loc=raw['localization'];targeted=raw['targeted']
    def find(n,f,score,method):return next(x for x in r['summary'] if (x['n'],x['family'],x['score'],x.get('method'))==(n,f,score,method))
    rows=[]
    for n in [1000,10000]:
        for f in ['tree','forest','logistic']:
            x=find(n,f,'shap','fixed3');comp=next(x for x in c['summary'] if (x['n'],x['family'],x['score'],x['method'])==(n,f,'shap','fixed3'))
            cross=[a['mean']*100 for a in x['aligned_warning'][1:]]
            rows.append([f'{n:,}',f.title(),f"{100*x['self_warning']['mean']:.2f}",f"{100*x['aligned_warning'][0]['mean']:.2f}",f'{min(cross):.2f}--{max(cross):.2f}',f"{100*comp['null_warning']['mean']:.2f}"])
    table('transfer_warning.tex','llrrrr',r'$n$ & Learner & Fixed $f$ & New fit, same family & New fit, other families & Composite, same family',rows)
    rows=[]
    for n in [1000,10000]:
        for f in ['tree','forest','logistic']:
            a,b=[find(n,f,k,'fixed3') for k in ['shap','flip']]
            rows.append([f'{n:,}',f.title(),a['harm_cases'],f"{a['laws'][3]['mean_F1_loss']:.3f}",f"{100*a['laws'][0]['harm_localization']['mean']:.1f}",f"{100*a['laws'][3]['harm_localization']['mean']:.1f}",f"{100*b['laws'][0]['harm_localization']['mean']:.1f}",f"{100*b['laws'][3]['harm_localization']['mean']:.1f}"])
    table('transfer_harm.tex','llrrrrrr',r'$n$ & Learner & $N_H$ & Mean F1 loss & SHAP, intact & SHAP, shifted & Flip, intact & Flip, shifted',rows)
    rows=[]
    for f in ['tree','forest','logistic']:
        entries=[find(10000,f,'shap',m) for m in r['methods']]
        rows.append([f.title(),f"{100*entries[1]['eta2_mean']:.1f}",*[f"{100*x['laws'][0]['all_localization']['mean']:.1f}" for x in entries]])
    table('transfer_bins.tex','lrrrrr',r'Learner & Variance explained & Global & Fixed bins & Source thirds & Source fifths',rows)
    rows=[]
    feature_names={'log_RMS':r'log RMS','kurtosis':'Kurtosis','raw_IR_1':r'IR raw, first harmonic','raw_OR_1':r'OR raw, first harmonic','band_4000_6001':r'4--6 kHz band','band_0_1000':r'0--1 kHz band'}
    for x in w['records']:
        setting='Original' if x['condition']=='original' else ('Resonance' if x['view']=='matched' else 'Removed')
        rows.append([setting,x['fault'],*[f"{x[k]:.3f}" for k in ['R','D','S']],feature_names[x['S_feature']],f"{x['band_share']:.3f}"])
    table('cwru_descriptive_audit.tex','llrrrlr',r'Acquisition & Head & $R_m$ & $D_m$ & $S_m$ & Maximizing outside feature & Band share',rows)
    # Harm localization and realized mean performance loss, for fixed fitted candidates.
    lines=[r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=3 by 2,horizontal sep=0.45cm,vertical sep=0.8cm},',
           r'scale only axis,width=0.275\textwidth,height=0.15\textwidth,xmin=0,xmax=1,xtick={0,0.25,0.5,0.75,1},',
           r'tick label style={font=\footnotesize,/pgf/number format/fixed},label style={font=\footnotesize},title style={font=\footnotesize},',
           r'grid=major,grid style={gray!15},legend columns=3,legend style={font=\footnotesize,draw=none},every axis plot/.append style={line width=0.8pt,mark size=1.8pt}]']
    for row in range(2):
        for col,f in enumerate(['tree','forest','logistic']):
            a=find(10000,f,'shap','fixed3');b=find(10000,f,'flip','fixed3');idx=[i for i,x in enumerate(r['records']) if x['n']==10000 and x['family']==f]
            harm=f1[idx,0].mean(axis=1)-f1[idx,3].mean(axis=1)>=.05
            opts=[f'title={{{f.title()}, $N_H={harm.sum()}$}}','ymin=0,ymax=1.02' if row==0 else 'ymin=0,ymax=0.8']
            if col==0:opts.append('ylabel={'+('Localization on harmful models' if row==0 else 'Mean F1 loss, all models')+'}')
            else:opts.append('yticklabels=\\empty')
            if row==1:opts.append('xlabel={Context bit-flip probability}')
            else:opts.append('xticklabels=\\empty')
            if row==0 and col==0:opts.append('legend to name=harm-legend')
            lines.append(r'\nextgroupplot['+','.join(opts)+']')
            if row==0:
                series=[('SHAP excess','black!80','*',[v['harm_localization']['mean'] for v in a['laws']]),
                        ('Flip excess','auditblue','square*',[v['harm_localization']['mean'] for v in b['laws']]),
                        ('Targeted context','auditorange','triangle*',targeted[idx].mean(axis=2)[harm].mean(axis=0))]
            else:series=[('Loss','black!80','*',[v['mean_F1_loss'] for v in a['laws']])]
            for label,color,marker,values in series:
                pairs=' '.join(f'({e:.17g},{v:.17g})' for e,v in zip(r['laws'],values))
                lines.append(r'\addplot+[color='+color+',mark='+marker+',mark options={fill='+color+r'}] coordinates {'+pairs+'};')
                if row==0 and col==0:lines.append(r'\addlegendentry{'+label+'}')
    lines +=[r'\end{groupplot}',r'\node[anchor=north] at ([yshift=-1.1cm]group c2r2.south) {\pgfplotslegendfromname{harm-legend}};',r'\end{tikzpicture}']
    (PAPER/'figures/transfer_harm.tex').write_text('\n'.join(lines)+'\n')
    # The first reference per learner is shown; score trends are exploratory diagnostics.
    old=np.load(OUT/'observed_overlap_draws.npz');counts=old['target_counts'];diag=[]
    lines=[r'\begin{tikzpicture}',r'\begin{axis}[width=0.93\columnwidth,height=0.57\columnwidth,',
           r'xmin=0.8,xmax=1,xtick={0.8,0.9,1},ymin=0,ymax=0.32,grid=major,grid style={gray!15},',
           r'xlabel={Observed overlap $O$},ylabel={Mean null score $A_f$},',
           r'legend style={font=\scriptsize,at={(0.02,0.98)},anchor=north west,draw=none},',
           r'tick label style={font=\footnotesize,/pgf/number format/fixed},label style={font=\footnotesize}]']
    for col,f in enumerate(['tree','forest','logistic']):
        item=next(x for x in r['records'] if x['n']==10000 and x['family']==f);i=item['array_index'];o=observed_overlap(counts[i]);a=scores(item['reference_table'],counts[i])[0]
        entries=[]
        for lo,hi in zip(np.linspace(.8,1,11)[:-1],np.linspace(.8,1,11)[1:]):
            use=(o>=lo)&((o<hi)|((hi==1)&(o<=hi)))
            if use.any():entries.append((float(o[use].mean()),float(a[use].mean())))
        color,marker=[('black','*'),('auditblue','square*'),('auditorange','triangle*')][col]
        lines.append(r'\addplot+[color='+color+',mark='+marker+r',line width=0.8pt] coordinates {'+' '.join(f'({x:.17g},{y:.17g})' for x,y in entries)+'};')
        lines.append(r'\addlegendentry{'+f.title()+'}')
        diag.append({'family':f,'case_seed':item['case_seed'],'points':entries})
    lines +=[r'\end{axis}',r'\end{tikzpicture}'];(PAPER/'figures/overlap_score.tex').write_text('\n'.join(lines)+'\n')
    summary={'diagnostic_points':diag,'hardware':w['environment'],'SHAP_seconds_per_576_windows':[x['tree_shap_seconds'] for p in w['proof'] if 'folds' in p for x in p['folds']],
             'exact_cluster_swap_p':w['finite_inventory']['swap_two_sided_p']}
    (OUT/'reference_transfer_assets.json').write_text(json.dumps(summary,indent=2)+'\n')
    durations=summary['SHAP_seconds_per_576_windows']
    runtime=f'{min(durations):.2f}--{max(durations):.2f}'
    macros=[r'\newcommand{\cwruShapRuntime}{'+runtime+'}',r'\newcommand{\cwruHardware}{'+w['environment']['cpu']+'}',r'\newcommand{\cwruShapVersion}{'+w['environment']['shap']+'}']
    (PAPER/'tables/reference_runtime.tex').write_text('\n'.join(macros)+'\n')
    print('Exported four result tables and two native PGFPlots figures')


if __name__=='__main__':main()
