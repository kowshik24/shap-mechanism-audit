"""Main-paper matched-size and proxy-strength assets; all study rows remain saved."""
import pdm_audit
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';PAPER=ROOT/'IEEE-conference-template-062824'


def coordinates(rows,x,y):
    return ' '.join(f'({x(r):.17g},{y(r):.17g})' for r in rows)


def write_plots(calibration,power):
    styles=[('global_target','Global','black!75','*'),
            ('observed_source','Observed/source','auditblue','square*'),
            ('observed_target','Observed/target','auditorange','triangle*')]
    for kind,data in [('size_power',calibration),('proxy_power',power)]:
        lines=[r'\begin{tikzpicture}',r'\begin{groupplot}[',
               r'group style={group size=3 by 2,horizontal sep=0.40cm,vertical sep=0.85cm},',
               r'scale only axis,width=0.278\textwidth,height=0.17\textwidth,',
               r'tick label style={font=\footnotesize,/pgf/number format/fixed,/pgf/number format/precision=2},label style={font=\footnotesize},',
               r'title style={font=\footnotesize},grid=major,grid style={gray!18},',
               r'ymin=0,ymax=1.02,ytick={0,0.2,0.4,0.6,0.8,1},',
               r'legend style={font=\footnotesize,draw=none,/tikz/every even column/.append style={column sep=7pt}},',
               r'legend columns=-1,',
               r'every axis plot/.append style={mark size=1.6pt,line width=0.8pt}]']
        for row,n in enumerate([1000,10000]):
            for col,family in enumerate(['tree','forest','logistic']):
                opts=[f'title={{{family.title()}, $n={n:,}$}}']
                if row==0 and col==0:opts+=['legend to name=legend-'+kind]
                if kind=='size_power':opts+=['xmin=0,xmax=0.21,xtick={0,0.05,0.10,0.15,0.20}']
                else:opts+=['xmin=0,xmax=0.5,xtick={0,0.1,0.2,0.3,0.4,0.5}']
                if row==1:opts+=['xlabel={'+('Null warning frequency' if kind=='size_power' else 'Context bit-flip probability')+'}']
                else:opts+=['xticklabels=\\empty']
                if col==0:opts+=['ylabel={'+('Context localization' if kind=='size_power' else 'Localization across all fits')+'}']
                else:opts+=['yticklabels=\\empty']
                lines.append(r'\nextgroupplot['+','.join(opts)+']')
                for method,label,color,mark in styles:
                    if kind=='proxy_power' and method=='observed_source':continue
                    rows=sorted([r for r in data['summary'] if (r['n'],r['family'],r['method'])==(n,family,method)],
                                key=lambda r:r['alpha'] if kind=='size_power' else r['error_probability'])
                    x=(lambda r:r['matched_warning_rate']['mean']) if kind=='size_power' else (lambda r:r['error_probability'])
                    y=(lambda r:r['matched_localization_rate']['mean']) if kind=='size_power' else (lambda r:r['all_cases']['mean'])
                    lines.append(r'\addplot+[color='+color+',mark='+mark+',mark options={solid,fill='+color+',draw='+color+r'}] coordinates {'+coordinates(rows,x,y)+'};')
                    if row==0 and col==0:lines.append(r'\addlegendentry{'+label+'}')
                    if kind=='size_power':
                        primary=[r for r in rows if r['alpha']==.05]
                        lines.append(r'\addplot[only marks,forget plot,color='+color+',mark='+mark+r',mark size=2.8pt] coordinates {'+coordinates(primary,x,y)+'};')
        lines += [r'\end{groupplot}',r'\node[anchor=north] at ([yshift=-1.20cm]group c2r2.south) {\pgfplotslegendfromname{legend-'+kind+'}};',r'\end{tikzpicture}']
        (PAPER/f'figures/review2_{kind}.tex').write_text('\n'.join(lines)+'\n')


def write_paired_tests(data):
    lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'$n$ & Learner & Bin only & Global only & Adjusted $p$ \\',r'\midrule']
    for n in [1000,10000]:
        for family in ['tree','forest','logistic']:
            r=next(r for r in data['paired_primary_tests'] if (r['n'],r['family'],r['comparison'])==(n,family,'observed_target'))
            if r['holm_p']==1:prob='$1.00$'
            else:
                exponent=int(f"{r['holm_p']:.2e}".split('e')[1]);mantissa=r['holm_p']/10**exponent
                prob=f'${mantissa:.2f}\\times10^{{{exponent}}}$'
            lines.append(f"{n:,} & {family.title()} & {r['conditional_only']} & {r['global_only']} & {prob}"+r' \\')
    lines +=[r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/review2_paired.tex').write_text('\n'.join(lines)+'\n')


def main():
    c=json.loads((OUT/'observed_overlap.json').read_text());p=json.loads((OUT/'proxy_power.json').read_text())
    s=json.loads((OUT/'cwru_spectral_transfer.json').read_text());assert c['complete'] and p['complete'] and s['complete']
    rows=[r for r in c['summary'] if r['alpha']==.05 and r['method'] in ['global_target','observed_target']]
    text=[r'\begin{tabular}{llrrrrr}',r'\toprule',
          r'$n$ & Learner & $N_C$ & Global FW & Bin FW & Global loc. & Bin loc. \\',r'\midrule']
    for n in [1000,10000]:
        for family,label in [('tree','Tree'),('forest','Forest'),('logistic','Logistic')]:
            a=next(r for r in rows if (r['n'],r['family'],r['method'])==(n,family,'global_target'))
            b=next(r for r in rows if (r['n'],r['family'],r['method'])==(n,family,'observed_target'))
            text.append(f"{n:,} & {label} & {a['material_case_n']} & {100*a['matched_warning_rate']['mean']:.2f} & {100*b['matched_warning_rate']['mean']:.2f} & {100*a['matched_localization_rate']['mean']:.1f} & {100*b['matched_localization_rate']['mean']:.1f}"+r' \\')
    text +=[r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/review2_matched.tex').write_text('\n'.join(text)+'\n')
    write_plots(c,p)
    write_paired_tests(c)
    spectral=s['results'][1]
    table=[r'\begin{tabular}{lrr}',r'\toprule',r'Training & Matched F1 & Filter removed F1 \\',r'\midrule',
           f"Original & {s['results'][0]['matched']['recording_macro_f1']:.3f} & {s['results'][0]['filter_removed']['recording_macro_f1']:.3f}"+r' \\',
           f"Resonance-confounded & {spectral['matched']['recording_macro_f1']:.3f} & {spectral['filter_removed']['recording_macro_f1']:.3f}"+r' \\',
           r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/review2_spectral.tex').write_text('\n'.join(table)+'\n')
    flat=[]
    for r in c['summary']:
        flat.append({'n':r['n'],'family':r['family'],'method':r['method'],'alpha':r['alpha'],
                     'null_warning':r['matched_warning_rate']['mean'],'localization':r['matched_localization_rate']['mean'],
                     'null_ci':r['matched_warning_rate']['case_bootstrap_95'],'localization_ci':r['matched_localization_rate']['case_bootstrap_95'],
                     'material_cases':r['material_case_n']})
    with (OUT/'review2_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print('COMPLETE: matched-size and proxy curves, primary tables, and complete summary CSV')


if __name__=='__main__':main()
