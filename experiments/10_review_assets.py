"""Generate review tables/figures from completed validation JSON and historical results.

No fitting or outcome selection occurs here. All added numeric text is emitted as LaTeX
macros from the same JSON that supplies tables and figures.
"""
import json
from pathlib import Path
import pdm_audit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pdm_audit.config import PHYSICAL
from pdm_audit.metrics import clopper_pearson

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'IEEE-conference-template-062824'
TABLES=TARGET/'tables'; FIGURES=TARGET/'figures'
R=ROOT/'experiments/results'

def read(name): return json.loads((R/name).read_text())
def write(name, text): (TABLES/name).write_text(text+'\n')

def main():
    v=read('review_validation.json')
    assert v.get('complete') and not v['quick'], 'Full validation is incomplete; do not publish partial/smoke results.'
    assert len(v['generated_rows'])==20 and len(v['negative_controls'])==100
    audit=read('audit.json'); sweep=read('shortcut_sweep.json'); scale=read('scale.json')
    # Keep the historical table body intact; the manuscript uses a four-mechanism view.
    lines=(TABLES/'audit.tex').read_text().splitlines()
    write('audit_physical.tex','\n'.join(line for line in lines if not line.startswith('RNF &')))
    lines=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Mode & Recall (95\% CI) & $L_m$ & Decoy max. & $R_m$ \\',r'\midrule']
    for m in PHYSICAL:
        q=v['oof_recall'][m];lo,hi=q['exact_95_interval'];e=v['reference_effects'][m]; d=v['feature_decoys'][m]
        lines.append(f"{m} & {q['recall']:.3f} ({lo:.3f}--{hi:.3f}) & {e['headroom_lift']:.3f} & {d['decoy_max']:.3f} & {d['R']:.3f} "+r"\\")
    write('review_effects.tex','\n'.join(lines+[r'\bottomrule',r'\end{tabular}']))
    labels={'historical':'Historical', 'exclude_OSF_comparison':'OSF excluded from comparison',
            'strict_single_both_cohorts':'Both cohorts single-flag', 'wear_representations':'Three wear representations',
            'wear_representations_no_OSF':'Grouped wear, OSF excluded'}
    lines=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'TWF definition & $n_m$/$n_o$ & $R$ & $D$ & $S$ \\',r'\midrule']
    for key,row in v['twf_sensitivity'].items():
        lines.append(f"{labels[key]} & {row['n_own']}/{row['n_other']} & {row['R']:.3f} & {row['D']:.3f} & {row['S']:.3f} "+r"\\")
    write('review_twf.tex','\n'.join(lines+[r'\bottomrule',r'\end{tabular}']))
    lines=[r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',
           r'& \multicolumn{2}{c}{100 label controls} & \multicolumn{2}{c}{10 uninjected seeds} & \multicolumn{2}{c}{10 injected seeds} \\',
           r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
           r'Mode & $R,D$ & $S$ & $R,D$ & $S$ & $R,D$ & $S$ \\',r'\midrule']
    for m in PHYSICAL:
        values=[]
        for summary in [v['negative_control_summary'],v['generated_summary']['None'],v['generated_summary']['0.25']]:
            for key in ['RD_support','S_warning']:
                q=summary[m][key];values.append(f"{q['count']}/{q['n']}")
        lines.append(m+' & '+' & '.join(values)+r' \\')
    write('review_rates.tex','\n'.join(lines+[r'\bottomrule',r'\end{tabular}']))
    # Metadata for exact finite-reference uncertainty on the historical largest case.
    largest=next(row for row in scale['rows'] if row['n']==300000 and row['sigma']==.25)
    adjusted=largest['p_holm_S']; order=sorted(adjusted,key=adjusted.get)
    # Its two strictly increasing leading Holm values uniquely identify these hit counts.
    assert order[:2]==['HDF','TWF'] and adjusted['HDF']<adjusted['TWF']<1
    largest_ci={}
    for rank,m in enumerate(order[:2]):
        raw=adjusted[m]/(4-rank); hits=round(raw*(scale['K']+1)-1)
        assert abs((hits+1)/(scale['K']+1)-raw)<1e-10
        largest_ci[m]={'hits':hits, 'raw_p':raw,'raw_tail_95_interval':list(clopper_pearson(hits,scale['K']))}
    controls = v['negative_controls']
    family_events = {}
    for event in ['RD_support', 'S_warning']:
        if event == 'RD_support':
            count = sum(any(max(row['families'][s][m]['p_holm'] for s in 'RD') < .05 for m in PHYSICAL) for row in controls)
        else:
            count = sum(any(row['families']['S'][m]['p_holm'] < .05 for m in PHYSICAL) for row in controls)
        family_events[event] = {'count': count, 'n': len(controls), 'exact_95_interval': list(clopper_pearson(count, len(controls)))}
    summary={'conditional_family_events':family_events, 'large_case_mc':largest_ci,
             'generated_mc_uncertain':{str(sig):{s:{m:sum(not row['families'][s][m]['mc_decision_stable'] for row in v['generated_rows'] if row['sigma']==sig)
                                for m in PHYSICAL} for s in 'RDS'} for sig in [None,.25]}}
    (R/'review_asset_summary.json').write_text(json.dumps(summary,indent=1))
    # Generate a few result macros to avoid manual transcription as runs change.
    macro=[]
    def command(name,value):macro.append('\\newcommand{\\'+name+'}{'+str(value)+'}')
    clean=v['generated_summary']['None']; injected=v['generated_summary']['0.25']
    command('CleanTWFWarnings',clean['TWF']['S_warning']['count'])
    command('InjectedHDFWarnings',injected['HDF']['S_warning']['count'])
    command('InjectedTWFWarnings',injected['TWF']['S_warning']['count'])
    command('CleanTWFHighWarnings',clean['TWF']['S_floor_counts']['0.05'])
    command('InjectedHDFHighWarnings',injected['HDF']['S_floor_counts']['0.05'])
    command('InjectedHDFMCUncertain',summary['generated_mc_uncertain']['0.25']['S']['HDF'])
    lo, hi = injected['HDF']['S_warning']['exact_95_interval']
    command('InjectedHDFWarningInterval',f'{lo:.3f}--{hi:.3f}')
    q=v['predictive_shortcut_control']
    command('ShortcutOriginalFone',f"{q['original']['f1']:.3f}")
    command('ShortcutShiftedFone',f"{q['shortcut_permuted_test']['f1']:.3f}")
    write('review_numbers.tex','\n'.join(macro))
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    rows=sweep['rows']; x=np.arange(len(rows)); sig=[r['sigma'] for r in rows]
    fig,axes=plt.subplots(2,1,figsize=(3.45,3.8),sharex=True,gridspec_kw={'height_ratios':[1.5,1]},layout='constrained')
    ax=axes[0];ax.plot(x,[r['sensor_share_HDF'] for r in rows],'o-',label='Injected sensor',color='#0072B2',markersize=3)
    ax.plot(x,[r['R_HDF'] for r in rows],'s-',label='Expected-feature share',color='#D55E00',markersize=3)
    ax.plot(x,[r['null_mean_R_HDF'] for r in rows],'--',label='Reference mean',color='.55')
    for i,r in enumerate(rows):
        if r['p_holm_R']['HDF']>=.05:ax.scatter(i,r['R_HDF'],s=40,facecolors='none',edgecolors='black',zorder=4)
        if r['p_holm_S']['HDF']<.05:ax.scatter(i,r['sensor_share_HDF'],s=35,marker='^',facecolors='none',edgecolors='black',zorder=4)
    ax.set(ylabel='HDF attribution share',ylim=(0,1));ax.legend(frameon=False,loc='upper right')
    for stat,color,mark in [('R','#D55E00','o'),('D','#009E73','s'),('S','#0072B2','^')]:
        axes[1].plot(x,[r[f'p_holm_{stat}']['HDF'] for r in rows],marker=mark,markersize=3,label=f'$p_{stat}$',color=color)
    axes[1].axhline(.05,color='black',linestyle=':',linewidth=.8)
    axes[1].set(yscale='log',ylabel='Adjusted tail comparison',xlabel='Shortcut noise (decreasing)',xticks=x,xticklabels=[str(z) for z in sig],ylim=(.005,1.4))
    axes[1].tick_params(axis='x',rotation=50,labelsize=7);axes[1].legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02))
    fig.savefig(FIGURES/'shortcut_sweep_review.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.45),layout='constrained')
    for ax,summary,title,n in [(axes[0],v['negative_control_summary'],'Conditional label controls',100),
                                (axes[1],clean,'Generated, no injected sensor',10)]:
        for offset,key,label,color in [(-.19,'RD_support','R and D support','#D55E00'),(.19,'S_warning','S warning','#0072B2')]:
            vals=[summary[m][key]['rate'] for m in PHYSICAL]
            lo=[summary[m][key]['exact_95_interval'][0] for m in PHYSICAL]
            hi=[summary[m][key]['exact_95_interval'][1] for m in PHYSICAL]
            ax.errorbar(np.arange(4)+offset,vals,yerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='o',markersize=4,capsize=3,label=label,color=color)
        ax.set(title=f'{title} ({n} repeats)',xticks=np.arange(4),xticklabels=PHYSICAL,ylabel='Observed frequency',ylim=(-.02,1.02))
        if n == 100: ax.set_ylim(-.003,.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.savefig(FIGURES/'review_validation.pdf',bbox_inches='tight');plt.close(fig)
    print('Generated review tables, result macros, and both diagnostic figures')

if __name__=='__main__':main()
