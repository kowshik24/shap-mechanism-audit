"""Build manuscript assets only from completed scientific extension results."""
import pdm_audit
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'IEEE-conference-template-062824'
TABLES, FIGURES = TARGET/'tables', TARGET/'figures'
RESULTS = ROOT/'experiments/results'


def read(name):
    r = json.loads((RESULTS/name).read_text()); assert r['complete'], name
    return r


def table(name, columns, header, rows):
    text = ['\\begin{tabular}{@{}'+columns+'@{}}', r'\toprule', header+r' \\', r'\midrule']
    text += [' & '.join(row)+r' \\' for row in rows]
    (TABLES/name).write_text('\n'.join(text+[r'\bottomrule',r'\end{tabular}'])+'\n')


def main():
    c, k, t, e = [read(name) for name in ['cwru.json', 'known_truth_overlap.json', 'temporal.json', 'effect_replication.json']]
    followup = read('cwru_followup.json')
    assert len(followup['records']) == 16
    assert len(c['inventory']) == 36 and len(t['records']) == 11 and len(e['records']) == 40
    rows = []
    for r in k['sizes']:
        a = r['aligned_calibrated_warning']; lo, hi = a['exact_95_interval']
        rows.append([f"{r['n']:,}", f"{r['threshold']:.3f}",
                     f"{r['aligned_fixed_floor_warning']['count']}/200", f"{a['count']}/200",
                     f"{lo:.4f}--{hi:.4f}", f"{r['misaligned_calibrated_warning']['count']}/200"])
    table('known_truth.tex', 'rrrrrr', r'$n$ & $q$ & Fixed & Calibrated & 95\% interval & Proxy', rows)
    rows = []
    for label, key in [('Original', 'original'), ('Gain-correlated', 'gain_correlated')]:
        r = c['conditions'][key]
        rows.append([label, f"{r['matched']['recording_macro_f1']:.3f}",
                     f"{r['gain_removed']['recording_macro_f1']:.3f}"])
    table('cwru_prediction.tex', 'lrr', 'Training condition & Matched F1 & Gain removed F1', rows)
    rows = []
    for m in ['IR', 'B', 'OR']:
        a, b = [c['conditions'][key]['audit'][m] for key in ['original','gain_correlated']]
        rows.append([m, f"{a['R']:.3f}", f"{b['R']:.3f}", f"{a['gain_share']:.3f}", f"{b['gain_share']:.3f}"])
    table('cwru_attribution.tex', 'lrrrr', r'Location & $R$ original & $R$ gain & RMS original & RMS gain', rows)
    rows = []
    for axis, label in [('load_hp','Load'), ('diameter_in','Diameter')]:
        for view, title in [('all','All 19'),('anchors','Harmonic anchors'),('broad','Broad features'),('RMS','Log RMS only')]:
            original = next(r for r in followup['records'] if r['axis'] == axis and r['view'] == view and not r['gain_correlated'])
            gain = next(r for r in followup['records'] if r['axis'] == axis and r['view'] == view and r['gain_correlated'])
            correct = sum(r['truth'] == r['prediction'] for r in gain['gain_removed']['recordings'])
            rows.append([label,title,f"{original['matched']['recording_macro_f1']:.3f}",
                         f"{gain['matched']['recording_macro_f1']:.3f}",
                         f"{gain['gain_removed']['recording_macro_f1']:.3f}",f'{correct}/36'])
    table('cwru_followup.tex','llrrrr',r'Held-out axis & Inputs & Original F1 & Gain F1 & Gain removed F1 & Correct after removal',rows)
    rows = []
    generated = t['records'][1:]
    temporal_summary = {}
    for key, label in [('baseline','Current inputs'), ('past_context','Past context')]:
        released = t['records'][0]['models'][key]
        values = [r['models'][key] for r in generated]
        f1 = [r['test_F1'] for r in values]
        detected = sum(r['mechanism_recall']['TWF']['detected'] for r in values)
        total = sum(r['mechanism_recall']['TWF']['total'] for r in values)
        rr = released['mechanism_recall']['TWF']
        rows.append([label, f"{released['test_F1']:.3f}", f"{rr['detected']}/{rr['total']}",
                     f"{np.mean(f1):.3f}", f"{detected}/{total}"])
        temporal_summary[key] = {'mean_F1': float(np.mean(f1)), 'TWF_detected': detected, 'TWF_total': total,
                                 'min_F1': min(f1), 'max_F1': max(f1)}
    table('temporal.tex', 'lrrrr', r'Inputs & Release F1 & TWF & Generated F1 & TWF', rows)
    rows, scale_summary = [], []
    for n in [10000,30000,100000,300000]:
        values = [r for r in e['records'] if r['n'] == n and r['sigma'] == .25]
        clean = [r for r in e['records'] if r['n'] == n and r['sigma'] is None]
        share = [r['observed']['HDF']['share[Shortcut sensor]'] for r in values]
        relevant = [r['observed']['HDF']['R'] for r in values]
        f1 = [r['metrics']['matched']['F1'] for r in values]
        shifted = [r['metrics']['shortcut_permuted']['F1'] for r in values]
        row = {'n': n, 'mean_sensor_share': float(np.mean(share)), 'sensor_range': [min(share),max(share)],
               'mean_R': float(np.mean(relevant)), 'R_range': [min(relevant),max(relevant)],
               'mean_F1': float(np.mean(f1)), 'mean_shifted_F1': float(np.mean(shifted)),
               'mean_clean_R': float(np.mean([r['observed']['HDF']['R'] for r in clean]))}
        scale_summary.append(row)
        rows.append([f'{n:,}', f"{row['mean_R']:.3f}", f"{row['mean_sensor_share']:.3f}",
                     f"{row['mean_F1']:.3f}", f"{row['mean_shifted_F1']:.3f}"])
    table('effect_replication.tex','rrrrr',r'$n$ & HDF $R$ & Sensor share & F1 & Shift F1',rows)
    plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':9, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(7.1,2.5))
    colors = ['#9d301c','#146a91','#366f37']; names=['Fixed $S>0.05$','Calibrated aligned','Calibrated proxy']
    for j, key in enumerate(['aligned_fixed_floor_warning','aligned_calibrated_warning','misaligned_calibrated_warning']):
        x = np.arange(3)+(j-1)*.2
        rates = np.array([r[key]['rate'] for r in k['sizes']]); bounds = np.array([r[key]['exact_95_interval'] for r in k['sizes']])
        axes[0].errorbar(x, rates, yerr=np.stack([rates-bounds[:,0],bounds[:,1]-rates]), fmt='o',
                         color=colors[j], capsize=2, label=names[j], markersize=4)
    axes[0].set(xticks=range(3), xticklabels=['1k','10k','30k'], xlabel='Rows per oracle dataset',
                ylabel='Warning frequency', ylim=(-.04,1.04), title='(a) Known-function evaluation')
    axes[0].axhline(.05,color='.5',ls=':',lw=.8)
    axes[0].legend(fontsize=6.5, loc='center right', frameon=False)
    n = [r['n'] for r in scale_summary]
    axes[1].plot(n,[r['mean_R'] for r in scale_summary], 'o-', color='#146a91',label='Expected share $R$')
    axes[1].plot(n,[r['mean_sensor_share'] for r in scale_summary], 's-', color='#9d301c',label='Shortcut share')
    for key, mean_key, color in [('R_range','mean_R','#146a91'), ('sensor_range','mean_sensor_share','#9d301c')]:
        means = np.array([r[mean_key] for r in scale_summary])
        bounds = np.array([r[key] for r in scale_summary])
        axes[1].errorbar(n, means, yerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),
                         fmt='none', color=color, capsize=2, alpha=.65)
    axes[1].set(xscale='log',xticks=n,xticklabels=['10k','30k','100k','300k'],ylim=(0,.8),
                xlabel='Rows per generated dataset',ylabel='Mean HDF attribution share',title='(b) Five generation seeds')
    axes[1].legend(fontsize=7,frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(w_pad=2)
    fig.savefig(FIGURES/'scientific_extensions.pdf',bbox_inches='tight'); plt.close(fig)
    (RESULTS/'scientific_asset_summary.json').write_text(json.dumps({'temporal': temporal_summary,
            'scale': scale_summary}, indent=2))


if __name__ == '__main__':
    main()
