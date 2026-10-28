"""Generate a native TikZ matrix from every directed source-control result."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save(path, content):
    """Preserve timestamps when generated content is unchanged."""
    if not path.is_file() or path.read_text() != content:
        path.write_text(content)


def paper_figure(controls):
    """Matrix highlighting and comparison-plot layouts inspired by local examples."""
    results = json.loads((ROOT/'experiments/results/kaist_diagnostics.json').read_text())
    lines = [r'\begin{tikzpicture}[font=\footnotesize]',
        r'\tikzset{loadcell/.style={draw=black!40,minimum width=1.0cm,minimum height=.68cm}}',
        r'\node[font=\small] at (1.8,3.82) {(a) Forest load-pair controls};',
        r'\node at (1.8,3.34) {Test load (Nm)};',
        r'\node[rotate=90] at (-.6,1.4) {Training load (Nm)};']
    for i,load in enumerate([0,2,4]):
        lines += [rf'\node at ({.6+i*1.2},2.92) {{{load}}};',
                  rf'\node at (-.13,{2.4-i}) {{{load}}};']
    for i,train in enumerate([0,2,4]):
        for j,test in enumerate([0,2,4]):
            if train == test:
                fill,value = 'black!12',r'\textemdash'
            else:
                rate = next(x for x in controls if (x['family'],x['training_load'],x['validation_load'])==('forest',train,test))['metrics']['healthy_FPR']*100
                fill,value = ('auditorange!30' if rate else 'white'),f'{rate:.1f}'
            lines.append(rf'\node[loadcell,fill={fill}] at ({.6+j*1.2},{2.4-i}) {{{value}}};')
    lines += [r'\node[align=center] at (1.8,-.50) {Healthy false positives (\%)\\All logistic load-pair rates: 0\%};',
        r'\begin{axis}[auditaxis,at={(8.8cm,.28cm)},anchor=south west,',
        r'width=6.5cm,height=2.7cm,xmin=-3,xmax=103,ymin=.55,ymax=4.45,',
        r'xtick={0,25,50,75,100},ytick={1,2,3,4},',
        r'yticklabels={{0+2 Nm: $P_F\geq.5$},{0+2 Nm: argmax},{0 Nm: $P_F\geq.5$},{0 Nm: argmax}},',
        r'xlabel={Healthy false positives (\%)},',
        r'title={(b) Twenty seeds at 4 Nm},',
        r'extra y ticks={},clip=false]']
    for sources, rule, center in [([0],'argmax',4),([0],'probability',3),([0,2],'argmax',2),([0,2],'probability',1)]:
        rows = [x for x in results['seed_results'] if x['source_loads_Nm']==sources and x['load_Nm']==4]
        assert len(rows) == 20
        values = [100*(x['metrics']['healthy_FPR'] if rule=='argmax' else x['fault_probability_threshold_healthy_FPR']) for x in rows]
        color,mark = ('auditblue','*') if rule=='argmax' else ('auditorange','diamond')
        coordinates = ' '.join(f'({x:.17g},{center+j:.17g})' for x,j in zip(values,np.linspace(-.15,.15,20)))
        lines.append(r'\addplot[only marks,color='+color+',mark='+mark+r',mark size=1.45pt] coordinates {'+coordinates+'};')
    lines += [r'\end{axis}',r'\end{tikzpicture}']
    save(ROOT/'IEEE-conference-template-062824/figures/measured_load_plot.tex','\n'.join(lines)+'\n')


def main():
    controls = json.loads((ROOT/'experiments/results/kaist_verification.json').read_text())['directed_controls']
    target = ROOT/'IEEE-conference-template-062824/figures/kaist_load_controls.tex'
    lines = [r'\begin{tikzpicture}[font=\footnotesize]',r'\tikzset{loadcell/.style={draw=black!45,minimum width=1.25cm,minimum height=.85cm}}']
    for family, shift, title in [('logistic',0,'Logistic regression'),('forest',6.2,'Random forest')]:
        lines += [rf'\begin{{scope}}[xshift={shift}cm]',rf'\node[font=\small] at (1.4,4.15) {{{title}}};',
                  r'\node at (1.4,3.65) {Test load (Nm)};',r'\node[rotate=90] at (-1.5,1.25) {Training load (Nm)};']
        for i,load in enumerate([0,2,4]):
            lines += [rf'\node at ({i*1.4},3.17) {{{load}}};',rf'\node at (-.95,{2.5-i*1.1}) {{{load}}};']
        for i,train in enumerate([0,2,4]):
            for j,test in enumerate([0,2,4]):
                if train == test:
                    fill,value = 'black!12',r'\textemdash'
                else:
                    row = next(r for r in controls if (r['family'],r['training_load'],r['validation_load']) == (family,train,test))
                    rate = row['metrics']['healthy_FPR']*100
                    fill,value = ('auditorange!30' if rate > 0 else 'white'),f'{rate:.1f}'
                lines.append(rf'\node[loadcell,fill={fill}] at ({j*1.4},{2.5-i*1.1}) {{{value}}};')
        lines.append(r'\end{scope}')
    lines += [r'\node[align=center,text width=12cm,anchor=north] at (4.5,-.65) {Healthy false-positive rate (\%); diagonal not evaluated.\\',
              r'Values collapse three-class argmax; the highlighted control gives 100\%\\',
              r'under a summed fault-probability threshold of 0.5. One normal record per load.};',r'\end{tikzpicture}']
    save(target,'\n'.join(lines)+'\n')
    paper_figure(controls)
    print('Generated native matrix: all twelve directed learner/load controls')


if __name__ == '__main__':
    main()
