"""Export one explicitly selected saved forest case as exact SHAP share plots."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import numpy as np
from pdm_audit.exact_oracle import STATES,exact_shap,statistics

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';PAPER=ROOT/'IEEE-conference-template-062824'


def main():
    study=json.loads((OUT/'observed_overlap.json').read_text())
    # Selection uses only family/size/order, never a score or warning outcome.
    row=next(r for r in study['records'] if r['n']==10000 and r['family']=='forest')
    learned=json.loads((OUT/'learned_calibration.json').read_text())
    model=next(r for r in learned['records'] if (r['n'],r['case_seed'],r['family'],r['role'])==(row['n'],row['case_seed'],'forest','evaluation'))
    counts=np.load(OUT/'observed_overlap_draws.npz')['test_counts'][row['array_index'],0]
    failure=STATES[:,:3].max(axis=1).astype(bool);own=STATES[:,0].astype(bool)
    groups=[own,failure&~own];records=[]
    text=[r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.2cm},',
          r'scale only axis,width=0.37\textwidth,height=0.13\textwidth,',
          r'ybar,/pgf/bar width=8pt,ymin=0,ymax=0.68,ytick={0,0.2,0.4,0.6},',
          r'xtick={0,1,2,3},xticklabels={$Z_0$,$Z_1$,$Z_2$,$C$},',
          r'tick label style={font=\footnotesize},title style={font=\footnotesize},label style={font=\footnotesize},',
          r'grid=major,grid style={gray!15},enlarge x limits=0.18,',
          r'legend columns=2,legend style={font=\footnotesize,draw=none,cells={anchor=west},/tikz/column sep=8pt}]']
    for view,title in [('mechanism_only','Reference $f$'),('unrestricted','Candidate $g$')]:
        phi=exact_shap(model['views'][view]['probability_table'],counts/counts.sum())
        shares=[]
        for mask in groups:
            mass=(abs(phi[mask])*counts[mask,None]).sum(axis=0)
            shares.append((mass/mass.sum()).tolist())
        stats=statistics(phi,counts)['0']
        records.append({'view':view,'shares':shares,'mechanism_0':stats})
        opts='title={'+title+'}'
        if view=='mechanism_only':opts+=',ylabel={Absolute-attribution share},legend to name=attribution-legend'
        else:opts+=',yticklabels=\\empty'
        text.append(r'\nextgroupplot['+opts+']')
        for k,(label,color) in enumerate([(r'Mechanism 0 ($\mathcal I_0$)','auditblue'),(r'Other failures ($\mathcal F\setminus\mathcal I_0$)','auditorange')]):
            pairs=' '.join(f'({j},{v:.17g})' for j,v in enumerate(shares[k]))
            text.append(r'\addplot+[draw='+color+',fill='+color+r'!65] coordinates {'+pairs+'};')
            if view=='mechanism_only':text.append(r'\addlegendentry{'+label+'}')
    text +=[r'\end{groupplot}',r'\node[anchor=north] at ([yshift=-0.65cm]group c1r1.south east) {\pgfplotslegendfromname{attribution-legend}};',r'\end{tikzpicture}']
    (PAPER/'figures/attribution_example.tex').write_text('\n'.join(text)+'\n')
    paths=[Path(__file__),OUT/'observed_overlap.json',OUT/'learned_calibration.json',OUT/'observed_overlap_draws.npz',ROOT/'src/pdm_audit/exact_oracle.py']
    output={'selection':'First n=10000 forest evaluation case in saved record order; first untouched test batch',
            'array_index':row['array_index'],'case_seed':row['case_seed'],'batch_index':0,'n':int(counts.sum()),
            'expected_set':['Z0'],'counts':counts.tolist(),'records':records,
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'explanation_figure.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Exported saved SHAP example:',row['array_index'],row['case_seed'])


if __name__=='__main__':main()
