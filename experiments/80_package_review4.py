"""Build a local reviewer archive from an explicit artifact allowlist."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts'


def main():
    OUT.mkdir(exist_ok=True)
    paths = {ROOT/x for x in ['pyproject.toml', 'requirements.txt', 'requirements-reproduce.txt', 'reproduce_score_design.sh', 'reproduce_calibration_scope.sh', 'reproduce_kaist_pilot.sh']}
    for folder, pattern in [('src', '*.py'), ('tests', '*.py'), ('experiments', '*.py'), ('experiments', '*PROTOCOL.md')]:
        paths.update((ROOT/folder).rglob(pattern) if folder in ['src', 'tests'] else (ROOT/folder).glob(pattern))
    paths.update(ROOT/'experiments'/x for x in ['REVIEW4_PREDICTIVE_BASELINE_AMENDMENT.md', 'MATCHED_POPULATION_PROTOCOL.md'])
    paths.add(ROOT/'experiments/KAIST_LOAD_PILOT_CONTRACT.json')
    paths.add(ROOT/'data/dtu_tep/metadata.json')
    paths.update(ROOT/'data/dataset_scout'/x for x in ['kaist_vibration_inventory.json',
        'kaist_acoustic_inventory.json', 'kaist_load_files.json', 'profile.txt', 'description.txt'])
    for name in ['reference_transfer.json', 'reference_transfer_draws.npz', 'composite_reference.json',
                 'observed_overlap_draws.npz', 'observed_overlap.json', 'learned_calibration.json', 'matched_population.json',
                 'matched_population_draws.npz', 'dtu_prospective_data.npz', 'dtu_prospective_source.json',
                 'dtu_dose_response.json', 'dtu_predictive_baselines.json', 'score_design.json',
                 'score_design_draws.npz', 'review4_plot_data.json', 'review4_verification.json',
                 'publication_figure_data.json', 'publication_figure_verification.json',
                 'review_audit.json', 'review_audit_draws.npz', 'review_audit_matched_grid.csv',
                 'dtu_prediction_replay.json', 'harm_screening.json', 'harm_screening_draws.npz',
                 'harm_screening_rates.csv', 'harm_screening_contrasts.csv', 'calibration_assets.json',
                 'calibration_scope_verification.json', 'calibration_sensitivity.json',
                 'calibration_sensitivity_draws.npz', 'logistic_crossover.json', 'dataset_feasibility.json',
                 'kaist_archive.json', 'kaist_signal_inventory.json', 'kaist_features.csv',
                 'kaist_predictions.csv', 'kaist_predictive.json', 'kaist_verification.json',
                 'kaist_diagnostics.json', 'kaist_diagnostic_arrays.npz',
                 'kaist_seed_predictions.csv', 'kaist_diagnostic_verification.json',
                 'submission_readiness.json', 'submission_cold_build.json']:
        paths.add(ROOT/'experiments/results'/name)
    paper = ROOT/'IEEE-conference-template-062824'
    paths.update(paper/x for x in ['IEEE-conference-template-062824.tex', 'IEEE-conference-template-062824.pdf',
                                 'IEEE-conference-template-062824.bbl', 'IEEEtran.cls', 'Makefile', 'refs.bib',
                                 'submission_metadata.json', 'REVIEW4_EVIDENCE_REPORT.md',
                                 'PUBLICATION_FIGURE_AUDIT.md', 'SECONDARY_MEASURED_STUDIES.tex.txt',
                                 'CALIBRATION_SCOPE_REVISION.md', 'CALIBRATION_TECHNICAL_DETAILS.tex.txt',
                                 'technical_refs.bib', 'FINAL_READINESS_AUDIT.md',
                                 'LOGISTIC_CROSSOVER_NOTE.md', 'REAL_SENSOR_DATASET_PLAN.md',
                                 'KAIST_LOAD_PILOT_REPORT.md', 'FINAL_CLAIM_AUDIT.md',
                                 'METHODS_REVISION.md', 'RESULTS_REVISION.md'])
    paths.update((paper/'tables').glob('*.tex'))
    paths.update((paper/'figures').glob('*.tex'))
    paths.add(paper/'figures/publication_figures.pdf')
    paths.update((paper/'figures').glob('*_standalone.pdf'))
    assert all(p.is_file() for p in paths)
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    pages = json.loads((ROOT/'experiments/results/review4_verification.json').read_text())['pdf_pages']
    readme = '''# Local reviewer reproducibility archive

This archive contains the revised FIGURE_DRAFT_PAGES-page submission-length manuscript, the full exploratory
score-design and DTU follow-up report, source code, protocols, saved model
probability tables, empirical training/audit distributions, process input
arrays, paired bootstrap draws, and local verification results. It excludes
Git history, credentials, raw data download caches, and historical review or
rebuttal files. It is prepared locally; it has not been published or uploaded.

## Reproduce the focused follow-up

Use Python 3.13 and install the project and requirements in a virtual environment:

    python3 -m venv .venv
    .venv/bin/python -m pip install -e . -r requirements-reproduce.txt
    PYTHON=.venv/bin/python bash reproduce_score_design.sh

For the calibration-scope revision from frozen inputs, use:

    PYTHON=.venv/bin/python bash reproduce_calibration_scope.sh

The pinned reproduction list contains only the focused Python runtime and test
dependencies, rather than the full historical research environment. It was
installed in a fresh Python 3.13.7 virtual environment outside the checkout.
The build also requires latexmk, pdflatex, bibtex, IEEEtran support, PGFPlots,
and Poppler (pdfinfo, pdftotext, pdffonts). Existing extracted DTU arrays are
included; no network data acquisition is required for this focused runner.
The original simulation release and extraction provenance are recorded in
experiments/results/dtu_prospective_source.json. Complete raw source archives
are not included. Historical experiment scripts are provided as source context;
their full data-dependent pipelines require the original acquisition steps.

The focused runner performs 42 score/partition comparisons with 400 paired
bank/case bootstrap replicates and 760 DTU dose-study fits, plus 60 predictive
baseline fits. Runtime and exact PDF pagination can vary with dependencies.
Experiments 75/76/79 calculate results; 77 generates manuscript assets; 81
generates six native TikZ/PGFPlots figures, standalone PDFs, and a review atlas.
Experiments 83/84/85/86 add output controls, full-grid diagnostics, prediction
replay of 300 existing IDV4-dose fits, and global harm screens at five rates and
both sizes with 400 paired bootstrap draws. Run reproduce_calibration_scope.sh
to regenerate this extension from its frozen inputs. Experiment 87 verifies its
360 rate cells, 120 paired contrasts, 3,600 decision comparisons, and 600 replay
records. Experiment 88 adds 200 fresh harm-label batches per candidate and ten
additional permutation streams; it changes no model, primary result, or boundary.
All 1,200 independently estimated harm labels agree with the primary labels.
Experiment 90 checks affine-logit identities on all 400 logistic candidates and
paired background/normalization contrasts from the existing bootstrap draws.
The companion KAIST pilot contains six measured-load final classifier results,
all twelve directed controls, source-only selection, source-background SHAP,
and forty seed fits. The single-source 0-Nm forest misclassifies 26/30 healthy
4-Nm windows. Two-source refits have zero collapsed-argmax healthy errors, but
probability-threshold decisions remain seed-sensitive. It does not identify a
causal acquisition shortcut or validate a calibrated warning. Native creation dates are
class-coupled, and independent bearing identities remain unidentified. Protocol,
features, predictions, hashes, replay checks, and a native TikZ plot are included.
Full KAIST/UCI archives are not included. Experiments 89/91 scout/check metadata;
92 downloads and authenticates the 2.66 GB vibration archive; 93/94 extract and fit
the pilot. Run reproduce_kaist_pilot.sh for the full pipeline (network required
on first use), or experiments 95/97 to replay saved results offline. Experiments
96/97 reconstruct probabilities and verify twenty seeds per training-coverage
set; experiment 98 generates the native TikZ control matrix.
Experiments 78/82/87 check hashes, ranks, summaries, tables, metadata, all
1,071 plot coordinates, 120 full-grid cells, nine measured-load cells, and local PDF requirements.
The six-page atlas includes the companion overlap, ablation, and dose plots;
three figures appear in the manuscript, including the measured-load plot.
Experiment 78 --enforce-page-limit passes at ten pages,
including references. IEEE type size and margins are unchanged; supporting
derivations and duplicated tables remain external technical assets, not an appendix.
The report distinguishes exploratory evidence, limited mild-bias sensitivity,
input use, realized harm, and the lack of physical field calibration.

The archive is a separate reproducibility artifact, not a paper appendix.
Acceptance of external artifacts must follow the session's submission rules.
No IEEE PDF eXpress validation or conference submission has been performed.
File hashes are listed in PACKAGE_MANIFEST.json.
The final claim audit and experiments/results/submission_readiness.json record
the isolated replay, dependency-closure repairs, and cold-build repair. The
Boolean model banks and raw-waveform extraction were not regenerated by that
focused audit; DTU prediction replay reconstructed 300 specified fits.
'''.replace('FIGURE_DRAFT_PAGES', str(pages))
    archive = OUT/'calibration_scope_reproducibility.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr('REPRODUCIBILITY_README.md', readme)
        z.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
        for p in sorted(paths):
            z.write(p, str(p.relative_to(ROOT)))
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any('RESPONSE' in x or '/.git/' in x or '__pycache__' in x for x in z.namelist())
    result = {'archive': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size,
              'files': len(paths)+2, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'published': False}
    (OUT/'calibration_scope_package_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
