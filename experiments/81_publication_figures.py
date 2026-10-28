"""Native TikZ/PGFPlots figures bound to frozen study artifacts; no new outcomes."""
import pdm_audit
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'
FIG = ROOT/'IEEE-conference-template-062824/figures'
FAMILIES = ['tree', 'forest', 'logistic']
DOSES = [.1, .25, .5, 1., 2.]


def read(name):
    return json.loads((OUT/name).read_text())


def coords(points):
    return ' '.join(f'({float(x):.17g},{float(y):.17g})' for x, y in points)


def line(points, options):
    return r'\addplot['+options+r'] coordinates {'+coords(points)+'};'


def dot_interval(x, y, lo, hi, color, mark='*', filled=True, vertical=False):
    """Draw explicit endpoints, with caps in typographic points, not data units."""
    if vertical:
        ends = [(x, lo), (x, hi)]
        cap = '-'
    else:
        ends = [(lo, y), (hi, y)]
        cap = '|'
    fill = color if filled else 'white'
    return [line(ends, f'no marks,color={color},line width=0.65pt,forget plot'),
            line(ends, f'only marks,color={color},mark={cap},mark size=2pt,forget plot'),
            line([(x, y)], f'only marks,color={color},mark={mark},mark size=2.3pt,mark options={{fill={fill},line width=0.8pt}},forget plot')]


def save(name, lines):
    path, content = FIG/name, '\n'.join(lines)+'\n'
    if not path.is_file() or path.read_text() != content:
        path.write_text(content)


def legend_marker(color, mark, label, filled=True):
    fill = color if filled else 'white'
    return [r'\addlegendimage{only marks,color='+color+',mark='+mark+',mark options={fill='+fill+'}}',
            r'\addlegendentry{'+label+'}']


def main():
    spec = importlib.util.spec_from_file_location('kaist_assets', ROOT/'experiments/98_kaist_assets.py')
    kaist_assets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaist_assets)
    kaist_assets.main()
    reference, composite = read('reference_transfer.json'), read('composite_reference.json')
    matched, design = read('matched_population.json'), read('score_design.json')
    dtu, profiles = read('dtu_dose_response.json'), read('review4_plot_data.json')
    exported = {'reference_transfer': [], 'matched_differences': [], 'specificity': [],
                'overlap_profiles': profiles, 'process': []}
    # 1. Same decisions, different reference targets. One size/partition, with CIs.
    lines = [r'\begin{tikzpicture}',
             r'\begin{axis}[auditaxis,width=0.66\textwidth,height=3.2cm,',
             r'xmin=0,xmax=13,xtick={0,2,4,6,8,10,12},ymin=0.45,ymax=3.55,',
             r'ytick={1,2,3},yticklabels={Logistic,Forest,Tree},',
             r'xlabel={Eligible-model warning frequency (\%)},',
             r'legend columns=3,legend style={at={(0.5,1.22)},anchor=south}]',
             line([(5, .45), (5, 3.55)], 'dashed,color=black!45,line width=0.6pt,forget plot')]
    specs = [('self', -.17, 'black!50', '*', 'Fixed model: new batches'),
             ('refit', 0., 'auditorange', 'square*', 'Fixed cut: new fits'),
             ('population', .17, 'auditblue', 'triangle*', 'Population cut: new fits')]
    for family, y in zip(FAMILIES, [3, 2, 1]):
        r = next(x for x in reference['summary'] if
                 (x['n'], x['family'], x['score'], x.get('method')) == (10000, family, 'shap', 'fixed3'))
        c = next(x for x in composite['summary'] if
                 (x['n'], x['family'], x['score'], x['method']) == (10000, family, 'shap', 'fixed3'))
        values = [r['self_warning'], r['aligned_warning'][0], c['null_warning']]
        for spec, value in zip(specs, values):
            key, delta, color, mark, _ = spec
            x = 100*value['mean']
            lo, hi = np.asarray(value['ci95'])*100
            lines += dot_interval(x, y+delta, lo, hi, color, mark)
            exported['reference_transfer'].append({'family': family, 'reference': key,
                'x': x, 'y': y+delta, 'ci': [lo, hi], 'mean': value['mean']})
    for _, _, color, mark, label in specs:
        lines += legend_marker(color, mark, label)
    lines += [r'\node[anchor=west,font=\footnotesize,text=black!65] at (axis cs:5.15,0.65) {Nominal 5\%};',
              r'\end{axis}', r'\end{tikzpicture}']
    save('reference_transfer_plot.tex', lines)
    # 2. Show the complete rate/size grid instead of selecting the 5%/10k slice.
    lines = [r'\begin{tikzpicture}',
             r'\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.25cm,vertical sep=1.5cm},',
             r'auditaxis,width=0.35\textwidth,height=3.0cm,xmin=0.5,xmax=5.5,ymin=0.5,ymax=6.5,',
             r'xtick={1,2,3,4,5},xticklabels={1,2.5,5,10,20},ytick={1,2,3,4,5,6},',
             r'xlabel={Matched warning rate (\%)},tick align=outside]']
    for law in range(2):
        for col, n in enumerate([1000, 10000]):
            cases = [x for x in matched['summary'] if x['n'] == n]
            labels = [x['family'].title()+(' / global' if x['method'] == 'global' else ' / bins') for x in cases]
            title = ['Before shift', 'Independent context'][law]+r', $n='+f'{n:,}'+r'$'
            opts = 'title={'+title+'},'
            if law == 0:
                opts += 'xlabel={},'
            opts += 'yticklabels={'+','.join(reversed(labels))+'}' if col == 0 else r'yticklabels=\empty'
            lines.append(r'\nextgroupplot['+opts+']')
            for j, row in enumerate(cases):
                for rate_index, rate in enumerate(matched['rates']):
                    value = 100*row['flip_minus_shap'][law][rate_index]
                    lo, hi = np.asarray(row['difference_ci95'][law][rate_index])*100
                    color = 'auditblue' if value < 0 else 'auditorange' if value > 0 else 'black'
                    x, y = rate_index+1, 6-j
                    lines.append(r'\path[fill='+color+r'!14,draw=white,line width=0.5pt] (axis cs:'+f'{x-.5},{y-.5}'+r') rectangle (axis cs:'+f'{x+.5},{y+.5}'+r');')
                    star = r'$^{*}$' if lo > 0 or hi < 0 else ''
                    label = f'{value:.1f}'+star
                    lines.append(r'\node[font=\footnotesize] at (axis cs:'+f'{x},{y}'+r') {'+label+r'};')
                    exported['matched_differences'].append({'n':n, 'family':row['family'], 'partition':row['method'],
                        'law':law, 'rate':rate, 'x':x, 'y':y, 'value_pp':value, 'ci':[lo,hi], 'label':label})
    lines += [r'\end{groupplot}', r'\end{tikzpicture}']
    save('matched_localization_plot.tex', lines)
    # 3. A compact comparison of score designs shows both harm groups, not power alone.
    keys = ['batch_share_failure', 'frozen_raw_failure', 'flip_failure', 'batch_targeted_C']
    names = ['Batch SHAP', 'Frozen raw SHAP', 'Flip excess', 'Targeted SHAP']
    lines = [r'\begin{tikzpicture}',
             r'\begin{groupplot}[group style={group size=3 by 1,horizontal sep=0.65cm},',
             r'auditaxis,width=0.235\textwidth,height=4.35cm,xmin=-3,xmax=103,',
             r'xtick={0,25,50,75,100},ymin=0.55,ymax=4.45,ytick={1,2,3,4},',
             r'xlabel={Context localization (\%)}]']
    for j, family in enumerate(FAMILIES):
        sample = next(x for x in design['summary'] if x['family'] == family)
        title = family.title()+f" ({sample['harm_cases']}/{sample['no_harm_cases']})"
        opts = 'title={'+title+'},'
        opts += 'yticklabels={'+','.join(reversed(names))+'}' if j == 0 else r'yticklabels=\empty'
        if j == 0:
            opts += r',legend to name=specificitylegend,legend columns=2'
        lines.append(r'\nextgroupplot['+opts+']')
        for k, key in enumerate(keys):
            row = next(x for x in design['summary'] if
                       (x['family'], x['score'], x['partition']) == (family, key, 'global'))
            for group, delta, color, fill in [(0, .1, 'auditblue', True), (1, -.1, 'auditorange', False)]:
                value = 100*row['matched_localization'][1][group]
                lo, hi = np.asarray(row['matched_ci95'][1][group])*100
                lines += dot_interval(value, 4-k+delta, lo, hi, color, filled=fill)
                exported['specificity'].append({'family': family, 'score': key, 'group': group,
                    'x': value, 'y': 4-k+delta, 'ci': [lo, hi]})
        if j == 0:
            lines += legend_marker('auditblue', '*', 'Harm-positive')
            lines += legend_marker('auditorange', '*', 'Below harm threshold', False)
    lines += [r'\end{groupplot}',
              r'\node[anchor=north] at ([yshift=-0.85cm]group c2r1.south) {\pgfplotslegendfromname{specificitylegend}};',
              r'\end{tikzpicture}']
    save('harm_specificity_plot.tex', lines)
    # 4. Separate panels prevent the overlapping family curves/bands from hiding each other.
    lines = [r'\begin{tikzpicture}',
             r'\begin{groupplot}[group style={group size=3 by 1,horizontal sep=0.8cm},',
             r'auditaxis,width=0.26\textwidth,height=3.75cm,xmin=0.8,xmax=1,ymin=0,ymax=0.35,',
             r'xtick={0.8,0.9,1},ytick={0,0.1,0.2,0.3},xlabel={Observed overlap $O$}]']
    for j, profile in enumerate(profiles):
        means = np.asarray(profile['reference_means'])
        assert means.shape == (200, 10)
        assert np.allclose(means.mean(axis=0), profile['center'])
        assert np.allclose(np.quantile(means, [.025, .975], axis=0), [profile['lower'], profile['upper']])
        opts = 'title={'+profile['family'].title()+'},'
        opts += r'ylabel={Null excess score $A_f$}' if j == 0 else r'yticklabels=\empty'
        lines.append(r'\nextgroupplot['+opts+']')
        x, lower, upper = [np.asarray(profile[k]) for k in ['x', 'lower', 'upper']]
        polygon = list(zip(x, lower))+list(zip(x[::-1], upper[::-1]))
        lines.append(r'\addplot[draw=none,fill=auditblue,fill opacity=0.16] coordinates {'+coords(polygon)+r'} \closedcycle;')
        lines.append(line(zip(x, profile['center']), 'color=auditblue,mark=*,mark size=1.7pt,line width=0.9pt'))
    lines += [r'\end{groupplot}', r'\end{tikzpicture}']
    save('overlap_profiles.tex', lines)
    # 5. The same intervention can be localized while its removal leaves F1 unchanged.
    lines = [r'\begin{tikzpicture}',
             r'\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.2cm,vertical sep=1.0cm},',
             r'auditaxis,width=0.35\textwidth,height=3.3cm,xmode=log,log basis x=2,',
             r'xmin=0.075,xmax=2.6,xtick={0.1,0.25,0.5,1,2},xticklabels={0.1,0.25,0.5,1,2}]']
    process = []
    for kind in ['common', 'IDV4']:
        item = {'kind': kind, 'scores': [], 'losses': [], 'mean_loss': []}
        for score in ['shap', 'response']:
            rows = [next(x for x in dtu['summary'] if (x['score'], x['kind'], x['dose'], x['view']) ==
                         (score, kind, dose, 'matched')) for dose in DOSES]
            item['scores'].append({'score': score, 'rates': [100*x['localizations']/60 for x in rows],
                                   'ci': (100*np.array([x['localization_ci95'] for x in rows])).tolist()})
        for dose in DOSES:
            losses = []
            for unit in dtu['units']:
                if unit['role'] != 'evaluation':
                    continue
                pair = [r for r in unit['records'] if r['kind'] == kind and r['dose'] == dose]
                matched_f1 = next(r['metrics']['F1'] for r in pair if r['view'] == 'matched')
                removed_f1 = next(r['metrics']['F1'] for r in pair if r['view'] == 'removed')
                losses.append(matched_f1-removed_f1)
            assert len(losses) == 60
            item['losses'].append(losses)
            item['mean_loss'].append(float(np.mean(losses)))
        process.append(item)
    for j, item in enumerate(process):
        opts = 'title={'+('(a) Offset on all faults' if j == 0 else '(b) Offset on IDV4 only')+r'},ymin=-3,ymax=105,ytick={0,25,50,75,100},xticklabels=\empty,'
        opts += r'ylabel={Context localization (\%)}' if j == 0 else r'yticklabels=\empty'
        if j == 0:
            opts += r',legend to name=doselegend,legend columns=2'
        lines.append(r'\nextgroupplot['+opts+']')
        for row, color, mark in zip(item['scores'], ['auditblue', 'auditorange'], ['*', 'square*']):
            positions = np.asarray(DOSES)*(.97 if row['score'] == 'shap' else 1.03)
            lines.append(line(zip(positions, row['rates']), f'color={color},no marks,forget plot'))
            if j == 0:
                lines += legend_marker(color, mark, 'SHAP excess' if row['score'] == 'shap' else 'Median response')
            for x, value, ci in zip(DOSES, row['rates'], row['ci']):
                # Small log-axis offsets distinguish overlapping error bars; observed doses remain the ticks.
                shifted = x*(.97 if row['score'] == 'shap' else 1.03)
                lines += dot_interval(shifted, value, ci[0], ci[1], color, mark, vertical=True)
    for j, item in enumerate(process):
        opts = 'ymin=-0.035,ymax=0.36,ytick={0,0.1,0.2,0.3},xlabel={Offset / training IQR},'
        opts += r'ylabel={F1 loss after offset removal}' if j == 0 else r'yticklabels=\empty'
        lines.append(r'\nextgroupplot['+opts+']')
        lines.append(line([(.075, 0), (2.6, 0)], 'dashed,color=black!45,forget plot'))
        points = []
        for x, losses in zip(DOSES, item['losses']):
            jitter = np.exp(np.linspace(-.038, .038, 60))
            points += list(zip(x*jitter, losses))
        lines.append(line(points, 'only marks,mark=*,mark size=0.8pt,color=black!40,opacity=0.35,forget plot'))
        lines.append(line(zip(DOSES, item['mean_loss']), 'color=black,mark=diamond*,mark size=2.1pt,line width=0.9pt'))
        if j == 1:
            lines.append(r'\node[anchor=center,font=\footnotesize,align=center] at (axis cs:0.43,0.17) {All 48 IDV4 fault rows positive\\in every fit and view\\F1 changes were zero};')
    lines += [r'\end{groupplot}',
              r'\node[anchor=north] at ([yshift=-0.95cm]group c1r2.south east) {\pgfplotslegendfromname{doselegend}};',
              r'\end{tikzpicture}']
    exported['process'] = process
    save('process_dose_plot.tex', lines)
    # Data and standalone review exports use the same native sources as the paper.
    captions = {
        'reference_transfer_plot': r'Probability-audit warning rates increased on decision-equivalent refits. SHAP excess used fixed overlap bins and $n=10{,}000$; refits used the same learner family. Each learner had 200 evaluated fits with 200 batches each. Whiskers are 95\% fitted-case bootstrap intervals; population-cut intervals condition on the realized calibration bank. The dashed line marks nominal 5\%.',
        'matched_localization_plot': r'Localization differences depend on learner, warning rate, and context law. All 120 cells show flip-minus-SHAP localization differences in percentage points across two sizes, two partitions, five matched warning rates, and two laws. Blue/negative favors SHAP; orange/positive favors flip. An asterisk marks a pointwise 95\% paired-bootstrap interval excluding zero; these intervals are not multiplicity-adjusted. Fixed bins use target-law knowledge. Matching was retrospective.',
        'harm_specificity_plot': r'Post-shift localization did not consistently separate the harm groups. Selected score designs used global population ranks at matched 5\% empirical eligible-model warnings and $n=10{,}000$. Filled/hollow points denote $H_i\geq0.05$/$H_i<0.05$; panel counts follow that order. Whiskers are pointwise 95\% paired-bootstrap intervals. Batch SHAP is normalized; frozen raw SHAP changes both background and normalization. Targeted SHAP knows $C$; complete ablations accompany the numerical data.',
        'overlap_profiles': r'Null attribution excess increased with mechanism overlap in every learner. Curves average within-interval means over 200 references at $n=10{,}000$. Shading spans the 2.5th--97.5th percentiles across those reference means; it is not a confidence band. The ten plotted overlap intervals summarize curves; calibration instead uses the stated global or three-bin rule.',
        'process_dose_plot': r'Bias localization varied with fault cohort; binary decisions were saturated for IDV4. Top: matched-view context localization in 60 resampled-training DTU fit/packet pairs, with exact 95\% binomial intervals conditional on the frozen rank-96 cut for each score. Horizontal offsets distinguish error bars. Bottom: gray points are paired matched-minus-removed F1 changes; black diamonds are means. Prediction replay found all 48 IDV4 fault rows positive in every matched/removed view at every dose, explaining the zero F1 changes. Horizontal jitter shows coincident packet values. Lines connect tested doses, without a fitted dose-response model. Scores were not retrospectively size-matched.',
        'measured_load_plot': r'Measured healthy-row failure rates varied with training set and decision rule. (a) Healthy false-positive rates for all six directed forest load pairs at the original seed; every logistic control was zero. Diagonal cells were not evaluated. (b) Each point denotes one of twenty forest seeds on the same 4-Nm normal recording; vertical jitter separates seeds. Labels specify source loads and the binary decision rule. Argmax collapses the three-class prediction; $P_F$ sums fault probabilities. Points are not independent-asset replicates or confidence intervals.',
    }
    files = ['reference_transfer.json', 'composite_reference.json', 'matched_population.json',
             'score_design.json', 'dtu_dose_response.json', 'review4_plot_data.json', 'dtu_prediction_replay.json',
             'kaist_verification.json', 'kaist_diagnostics.json', 'kaist_diagnostic_verification.json']
    paths = [OUT/name for name in files]+[Path(__file__), FIG/'publication_style.tex', ROOT/'experiments/98_kaist_assets.py']
    result = {'complete': True, 'data': exported, 'captions': captions,
              'scope': 'Visualization of saved results only; no new model fits or outcomes',
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'publication_figure_data.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    paper = FIG.parent/'IEEE-conference-template-062824.tex'
    text = paper.read_text()
    for name, caption in captions.items():
        first, rest = caption.split('. ', 1)
        pattern = r'(\\input\{figures/'+re.escape(name)+r'\.tex\}\s*)\\caption\{.*?\}(\s*\\label\{[^}]+\})'
        replacement = r'\caption{\textbf{'+first+'.} '+rest+'}'
        text = re.sub(pattern, lambda m: m[1]+replacement+m[2], text, flags=re.S)
    if text != paper.read_text():
        paper.write_text(text)
    # A six-page review atlas; individual standalone wrappers are also exported.
    preamble = [r'\documentclass[10pt]{article}', r'\usepackage[letterpaper,margin=0.6in]{geometry}',
                r'\usepackage{times,amsmath,amssymb,pgfplots,caption}',
                r'\usepgfplotslibrary{groupplots}', r'\usetikzlibrary{arrows.meta,positioning}',
                r'\pgfplotsset{compat=1.18}', r'\definecolor{auditblue}{RGB}{33,113,181}',
                r'\definecolor{auditorange}{RGB}{179,88,6}', r'\input{publication_style.tex}',
                r'\pagestyle{empty}', r'\begin{document}']
    atlas = preamble.copy()
    for name, caption in captions.items():
        # The atlas is self-contained, so its table cross-reference is replaced by a description.
        local_caption = caption.replace(r'Table~\ref{tab:score-design}', 'the manuscript score-design table')
        title, description = local_caption.split('. ', 1)
        atlas += [r'\begin{figure}[!ht]', r'\centering', r'\input{'+name+'.tex}',
                  r'\caption{\textbf{'+title+'.} '+description+'}', r'\end{figure}', r'\clearpage']
        wrapper = [r'\documentclass[tikz,border=7pt]{standalone}', *preamble[2:9],
                   r'\setlength{\textwidth}{7.16in}', r'\begin{document}',
                   r'\input{'+name+'.tex}', r'\end{document}']
        save(name+'_standalone.tex', wrapper)
    atlas += [r'\end{document}']
    save('publication_figures.tex', atlas)
    print('Generated six native TikZ/PGFPlots figures, saved data bindings, and review atlas sources')


if __name__ == '__main__':
    main()
