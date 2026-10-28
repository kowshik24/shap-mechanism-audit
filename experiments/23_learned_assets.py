"""Generate learned-calibration manuscript assets from complete verified records."""
import pdm_audit
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'IEEE-conference-template-062824'
RESULTS = ROOT/'experiments/results'


def main():
    learned = json.loads((RESULTS/'learned_calibration.json').read_text())
    fixed = json.loads((RESULTS/'fixed_learner_calibration.json').read_text())
    region = json.loads((RESULTS/'region_budget.json').read_text())
    assert region['complete'] and region['oracle_region_metadata'] and len(region['records']) == 1200
    assert learned['complete'] and not learned['quick'] and len(learned['records']) == 2400
    assert fixed['complete'] and fixed['eligible_functions'] == 1200 and fixed['calibration_datasets'] == 240000
    names = {'tree':'Tree','forest':'Forest','logistic':'Logistic'}
    rows = [r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',
            r'Learner & $n$ & Base & High, all & High, region & Region 95\% interval & Full $C$: all; region \\',r'\midrule']
    for r in region['summary']:
        base = next(x for x in fixed['summary'] if (x['n'],x['family']) == (r['n'],r['family']))
        a,b = r['uniform_full_context'],r['targeted_full_context'];lo,hi=r['targeted_warning']['exact_95_interval']
        values = [names[r['family']],f"{r['n']:,}",f"{base['within_function']['count']}/200",
                  f"{r['uniform_warning']['count']}/200",f"{r['targeted_warning']['count']}/200",
                  f'{lo:.3f}--{hi:.3f}',
                  f"{a['count']}/{a['n']}; {b['count']}/{b['n']}"]
        rows.append(' & '.join(values)+r' \\')
    (TARGET/'tables/learned_calibration.tex').write_text('\n'.join(rows+[r'\bottomrule',r'\end{tabular}'])+'\n')
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42})
    fig,axes = plt.subplots(1,2,figsize=(7.1,2.65))
    families = ['tree','forest','logistic'];colors={1000:'#146a91',10000:'#a83b24'}
    for n in [1000,10000]:
        for ref,marker,offset,title in [('uniform','o',-.05,'All batches'),('targeted','^',.05,'High region')]:
            x = np.arange(3)+(-.13 if n == 1000 else .13)+offset
            for axis,key in [(axes[0],ref+'_warning'),(axes[1],ref+'_full_context')]:
                records = [next(r for r in region['summary'] if r['n'] == n and r['family'] == family)[key] for family in families]
                rates = np.array([r['rate'] for r in records]);bounds=np.array([r['exact_95_interval'] for r in records])
                axis.errorbar(x,rates,yerr=np.stack([rates-bounds[:,0],bounds[:,1]-rates]),fmt=marker,
                              color=colors[n],markersize=4,capsize=2,label=f'{title}, {n//1000}k')
    axes[0].axhline(.05,color='.5',lw=.8,ls=':')
    axes[0].set(ylim=(0,.46),ylabel='Warning frequency on eligible fits',title='(a) Fresh high-overlap batches')
    axes[0].legend(fontsize=6.5,loc='upper left',frameon=False,ncol=2)
    axes[1].set(ylim=(0,1),ylabel='Correct context localization',title='(b) Material context dependence')
    for ax in axes:
        ax.set_xticks(range(3),[names[f] for f in families]);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(w_pad=2)
    fig.savefig(TARGET/'figures/learned_calibration.pdf',bbox_inches='tight');plt.close(fig)


if __name__ == '__main__':
    main()
