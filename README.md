# Calibration Scope in Mechanism-Level SHAP Audits

Current revision: October 26, 2026. The manuscript is a controlled study of
probability-attribution calibration scope. It now includes an exact hard-output
control, the full matched-rate localization grid, known-context permutation F1
screening, unknown-feature warning comparisons, and a prediction replay explaining
DTU IDV4 saturation. The new global screens use all five rates and both sample
sizes with 400 paired bootstrap draws. See the [revision report](IEEE-conference-template-062824/CALIBRATION_SCOPE_REVISION.md)
and [current manuscript](IEEE-conference-template-062824/IEEE-conference-template-062824.pdf).

Run `PYTHON=.venv/bin/python bash reproduce_calibration_scope.sh` from the saved
reference/process assets. Experiments 83--87 calculate and verify the extension.
The isolated final audit used CPython 3.13.7 and the smaller pinned
`requirements-reproduce.txt` (`pip install -e . -r requirements-reproduce.txt`).
Six scientific JSONs and 72 regenerated arrays agreed, all 184 declared
dependency hashes resolved, and nine isolated tests passed. The cold-build
Makefile and generator timestamp issues found in that audit are fixed. See
[final claim audit](IEEE-conference-template-062824/FINAL_CLAIM_AUDIT.md).
Full exports contain 360 screen-rate cells, 120 paired contrasts, and 120
localization-grid cells. Native TikZ sources and a six-page figure atlas are in
the manuscript figure directory; three figures appear in the paper. The new
measured-load figure combines a highlighted control matrix and all seed outcomes.
Numerical,
PDF, and figure verifiers pass. The draft now has ten pages including references, with no appendix. Experiment
78 `--enforce-page-limit` passes and is enabled in the reproduction drivers.
Experiment 88 adds independent harm-label and permutation-randomization checks;
see [final readiness audit](IEEE-conference-template-062824/FINAL_READINESS_AUDIT.md). Local checks do not establish PDF eXpress
validation, external scientific validation, or field readiness.

The official special-session CFP is at
https://bigdataieee.org/BigData2026/calls/special-tuai/ and lists an October 30
submission deadline. The inspected SP11 portal lists October 31 AoE; plan for
the earlier October 30 unless organizers clarify. SP11 and its blank form were
verified October 26; no paper was submitted. The repository remains private;
reviewer access still requires an allowed artifact release. The local archive is
`artifacts/calibration_scope_reproducibility.zip`; it is not published or uploaded.

The [KAIST measured-load pilot](IEEE-conference-template-062824/KAIST_LOAD_PILOT_REPORT.md) is complete and integrated into the abstract, methods, results, limitations, and conclusion: 21 native vibration recordings, three held-out loads, six source-selected final results, and all twelve directed learner/load controls. Two-source fits had binary balanced accuracy 1.000 under collapsed argmax. The 0-Nm-only forest misclassified 26/30 healthy 4-Nm windows; a twenty-seed diagnostic retained 63.3--100% healthy errors. Probability-threshold decisions differ and are reported separately. Source-background TreeSHAP reconstructions and forty seed fits are verified. Native creation dates are class-coupled and independent bearings remain unidentified; calibrated warning sensitivity is not established. Full details and two additional pilot figures remain companion assets. Reproduce with `PYTHON=.venv/bin/python bash reproduce_kaist_pilot.sh` (first run downloads 2.66 GB). See [dataset decision and pilot plan](IEEE-conference-template-062824/REAL_SENSOR_DATASET_PLAN.md) for alternatives. The [logistic note](IEEE-conference-template-062824/LOGISTIC_CROSSOVER_NOTE.md) quantifies background/normalization effects on the controlled models.

Earlier revision context follows:
The [main paper](IEEE-conference-template-062824/IEEE-conference-template-062824.pdf)
now tests reference transfer, harmful context shift, non-SHAP baselines,
source-derived bins, matched acceptable-model warning curves, and descriptive
TreeSHAP audits on measured bearing data, including original mill records. A prospective
53-input DTU process study now tests 100 calibration and 60 new evaluation
model/run-packet pairs under an explicit simulation/training-recipe law.

Fixed-function self-calibration does not control warnings on independently
fitted aligned models. SHAP excess can miss context dependence after that
context decouples, even when F1 degrades. Composite calibration has a broader,
but explicit, training-population null. The paper reports these limitations
rather than claiming a general SHAP sensitivity or physical-validity guarantee.

[Review response](IEEE-conference-template-062824/REVIEW3_RESPONSE.md) ·
[Readiness assessment](IEEE-conference-template-062824/SUBMISSION_READINESS.md)

Run `PYTHON=.venv/bin/python bash reproduce_reference_study.sh` to regenerate
new studies and the current paper. Experiment 45 checks scientific arithmetic
and provenance; experiment 49 independently reconstructs the matched-rate study
and its joint calibration-bank/case bootstrap; experiment 78 checks current manuscript
bindings (experiment 41 remains a historical verifier). The pipeline also runs the continuous-process study (59–61), numerical
replay (63/65), and process assets (64). The field-only
runner is `PYTHON=.venv/bin/python bash reproduce_field_study.sh`; its first run
downloads 835 MB of original public SCA files and checks publisher hashes. The
[field report](IEEE-conference-template-062824/SCA_FIELD_REPORT.md) distinguishes
prediction errors, descriptive attribution, and missing within-case comparison
cohorts. Run `PYTHON=.venv/bin/python bash reproduce_process_study.sh` for the
prospective process study alone; the first extraction retains about 346 MB of complete
selected trajectories through source byte ranges. The [process report](IEEE-conference-template-062824/DTU_PROSPECTIVE_REPORT.md)
reports 0/60 unaltered SHAP warnings (95% interval extends to 5.96%), bias
localization, transfer loss, and the majority-class baseline. The [public-data assessment](IEEE-conference-template-062824/PUBLIC_DATASET_ASSESSMENT.md)
also audits CARE metadata. These studies do not establish physical field calibration. Earlier numbered studies below remain archived project
context; they are not extra evidence for the revised central claim.

The physical follow-up authenticates complete author-hosted Causal Chambers
archives: 57 acquisition sequences and 27,749 measurements. The published
randomization schedule replays for all 29 light records and three wind records;
other discrepancies are retained. Every one of 627 intervention/channel pairs
is assessed, with 70 locally supported responses and no invented negative
causal exclusions. See the [physical evidence report](IEEE-conference-template-062824/PHYSICAL_CAUSAL_MAP_REPORT.md)
and [acquisition design](IEEE-conference-template-062824/PHYSICAL_ACQUISITION_PLAN.md).
Run `PYTHON=.venv/bin/python bash reproduce_physical_map.sh` for the complete
source/analysis/verifier workflow. Physical new-machine independence and an
exhaustive causal map remain unverified; the paper now distinguishes supported,
excluded, and unresolved features.

The incomplete-map extension derives coordinatewise sharp R/D/S ranges and a
conservative rank rule using upper calibration/lower candidate scores. It
reconstructs 715 records without refitting, independently enumerating 2,002,176
compatible maps. Unknown features stay unknown; this is partial identification,
not a causal or sampling confidence certificate. See the [partial-map report](IEEE-conference-template-062824/PARTIAL_MAP_REPORT.md).
Run `PYTHON=.venv/bin/python bash reproduce_partial_map.sh` after the source
studies. Its runtime-compatibility check uses the `uv` CLI with `--no-project
--isolated` and pinned Python 3.11/NumPy 1.26.3/pandas 2.1.4; the main project
remains on Python 3.13. The checked legacy runtime generates identical wind
protocols, preserving the unresolved source associations. The [provenance request](IEEE-conference-template-062824/PHYSICAL_PROVENANCE_REQUEST.md)
is prepared and unsent.

Dataset: [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(also in `dataset/ai4i2020.csv`) — 10,000 synthetic industrial-style sensor rows,
3.39% failure rate. This is not a collection of physical equipment measurements.

## What was tried

| Stage | Techniques |
|---|---|
| Feature engineering | `Power`, `Temp_diff`, `Overstrain`, `Overstrain_ratio` — derived straight from the failure-mode physics, not generic polynomial features |
| Imbalance handling | SMOTE vs BorderlineSMOTE vs ADASYN, compared on validation F1; winner carried through |
| Dimensionality check | PCA vs t-SNE vs UMAP (diagnostic only — see below) |
| Models | Logistic Regression, SVM (RBF), KNN, Decision Tree, Random Forest, XGBoost, PyTorch MLP |
| Tuning | `RandomizedSearchCV` (F1-scored) + validation-set decision-threshold tuning on the winning model |
| Validation | 5-fold stratified CV (single 1,000-row test split isn't enough to trust on its own) |
| Calibration | Brier score and quantile-binned expected calibration error on saved test probabilities |
| Explainability | SHAP `TreeExplainer`, cross-checked against ground-truth TWF/HDF/PWF/OSF/RNF labels |
| Exploratory notebook only | Multi-label model predicting the five flags; not part of the numbered audit pipeline |

## Results

Best model: **tuned XGBoost**, BorderlineSMOTE-resampled training set, custom decision threshold
instead of the default 0.5.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **XGBoost (tuned)** | **1.00** | 0.79 | **0.89** | 0.988 |
| XGBoost (baseline) | 0.86 | 0.88 | 0.87 | 0.988 |
| Random Forest | 0.79 | 0.88 | 0.83 | 0.989 |
| PyTorch MLP | 0.65 | 0.91 | 0.76 | 0.988 |
| Decision Tree | 0.68 | 0.82 | 0.75 | 0.927 |
| SVM (RBF) | 0.42 | 0.88 | 0.57 | 0.976 |
| KNN | 0.43 | 0.79 | 0.56 | 0.877 |
| Logistic Regression | 0.24 | 0.85 | 0.37 | 0.943 |

### PCA vs t-SNE vs UMAP

Three views of the same 12-dimensional training data. PCA (linear) shows the classes smeared
together with no clean separating direction. t-SNE and UMAP (both nonlinear, preserve local
neighborhoods) show the failure points clumping into several distinct pockets rather than one
region — consistent with `Machine failure` actually being five separate physical mechanisms, each
carving out its own little neighborhood in feature space instead of one shared "failure zone".

![PCA vs t-SNE vs UMAP](assets/images/dimred_comparison.png)

Numbers above are from `experiments/01_benchmark.py` on the 1,000-row test split (34 failures, so
the tuned model's precision of 1.00 has a 95% Clopper–Pearson interval of 0.87–1.00). 5-fold CV of the
whole pipeline (Borderline-SMOTE inside the folds, default threshold): F1 = 0.822 ± 0.013,
macro-F1 = 0.908 ± 0.007. Hyperparameter search also resamples inside its folds (search CV F1 0.831). The notebook's older CV cell used plain SMOTE and reported 0.784; the
script is the reference.

Class imbalance is why accuracy is useless here — a model that predicts "no failure" every time
still scores ~96% accuracy. Precision/recall/F1 on the failure class is what actually matters.

<img src="assets/images/class_imbalance.png" width="420"> <img src="assets/images/pca_2d.png" width="420">

<img src="assets/images/roc_curves.png" width="420"> <img src="assets/images/pr_curves.png" width="420">

### Resampling, visualized

Same comparison [imbalanced-learn's own docs](https://imbalanced-learn.org/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html)
run — a plain Logistic Regression fit on the 2D PCA projection, once per sampler. Without
resampling the boundary barely bends toward the minority (red) class; each oversampler drags it
further into majority territory, trading precision for recall — the same tradeoff the metrics
table above shows in numbers.

![resampling comparison](assets/images/resampling_scatter_2d.png)
![decision boundary comparison](assets/images/decision_boundary_2d.png)

## Mechanism-level attribution diagnostics

The documented target combines five mechanisms, although the released flags contain
inconsistencies; the ordinary classifier is not told which mechanism fired. For
each mechanism the audit computes out-of-sample SHAP (5-fold cross-fitting) and three statistics:

- **relevant share** R — fraction of attribution on the mechanism's rows that falls on its trigger
  features;
- **specificity** D — how much larger that share is on the mechanism's rows than on other failures;
- **outside-set excess** S — the largest excess share any outside-set feature takes on
  the mechanism's rows. This motivates inspection; another genuine overlapping
  mechanism can produce a large score even for a perfect predictor.

Each statistic is compared with 1,000 copies of the fixed-settings pipeline retrained on permuted
labels, with separate Holm adjustment over four mechanisms. This tests enrichment above a
shuffled-label reference, not correctness of the failure mechanism. Model selection is not nested.

| Mechanism | Top out-of-sample SHAP feature | R | D | Verdict |
|---|---|---|---|---|
| TWF (tool wear) | `Tool wear [min]` | 0.19 | 0.13 | fails both (p = 0.20, 0.077) |
| HDF (heat dissipation) | `Rotational speed [rpm]` | 0.71 | 0.43 | passes (p = 0.004) |
| PWF (power) | `Power [W]` | 0.70 | 0.33 | passes (p = 0.004) |
| OSF (overstrain) | `Overstrain_ratio` | 0.67 | 0.42 | passes (p = 0.004) |
| RNF (random) | no dominant feature | — | — | descriptive only |

HDF, PWF, and OSF have historical reference support across five seeds and the tested
model/explainer variants; raw features change the HDF conclusion. TWF remains
sensitive to representation and explainer choice, with cross-fitted recall only 3/46.
At 300k rows, relevant-share support coexists with 27% attribution on a known proxy.
Its historical outside-set warning is based on only one exceedance in 200 draws
and is Monte Carlo-sensitive. New paired generated audits warn on 8/10 strong
injections; 9/10 HDF warning decisions have Monte Carlo bounds crossing 0.05.
Neither reference support nor warning absence certifies mechanism correctness.

![audit overview](paper/figures/audit_overview.png)

<img src="assets/images/shap_summary.png" width="420"> <img src="paper/figures/shap_crosscheck_heatmap.png" width="420">

## Running it

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name glowing-broccoli-venv --display-name "Python 3 (venv)"
```

Open `notebooks/Exploratory-Data-Analysis.ipynb`, select the `Python 3 (venv)` kernel, run all.
Headless equivalent:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=glowing-broccoli-venv \
  Exploratory-Data-Analysis.ipynb
```

Takes 15-20 minutes — the hyperparameter search alone is ~200 XGBoost fits.

**Apple Silicon note:** numpy/sklearn/xgboost/torch each bundle their own threaded BLAS/OpenMP
runtime, and running them together in one Jupyter kernel segfaults the kernel with no useful
traceback. The notebook forces everything single-threaded before any other imports
(`OMP_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE`) — don't strip that cell out. If xgboost itself
won't import (`libomp.dylib` not found), `brew install libomp` first.

## Scoring new data

The trained model and preprocessor are committed in `models/` (`experiments/01_benchmark.py`
regenerates them):

```bash
python src/predict.py --csv path/to/new_readings.csv --out predictions.csv
```

Input needs: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`,
`Torque [Nm]`, `Tool wear [min]`.

## Reproducing the paper

The code in `src/pdm_audit/` and the numbered scripts in `experiments/` produce the result
artifacts, tables, and figures for the revised IEEE paper
(`IEEE-conference-template-062824/IEEE-conference-template-062824.tex`). The notebook is the exploratory record and is not needed.

```bash
./reproduce.sh          # multi-hour run; builds IEEE-conference-template-062824/IEEE-conference-template-062824.pdf
./reproduce.sh --quick  # smoke test of the audit stages with tiny K
PYTHON=.venv/bin/python bash reproduce_scientific.sh  # completed scientific extensions only
```

or stage by stage:

| Stage | Script | Output |
|---|---|---|
| 1 | `experiments/01_benchmark.py` | model comparison, tuning, CIs, CV, `models/` |
| 2 | `experiments/02_audit.py` | cross-fitted audit, seeds, raw-feature and in-sample variants (~28 min) |
| 3 | `experiments/03_shortcut_sweep.py` | graded shortcut experiment (~11 min) |
| 4 | `experiments/04_tables.py` | LaTeX tables in `paper/ieee_bigdata/tables/` |
| 5 | `experiments/05_figures.py` | figures in `paper/figures/` |
| 6 | `experiments/06_explainers.py` | interventional backgrounds 100/500 at three seeds, 1,000 at seed 0; random forest (~55 min) |
| 7 | `experiments/07_scale.py` | audit on AI4I generated at 10k–300k rows (3 seeds), cost at 1M (about 7 h in the saved run) |
| 8 | `experiments/08_compare.py` | same-model subgroup and in-sample comparisons on shortcut and shuffled-label controls |
| 9 | `experiments/09_review_validation.py` | prediction-preserving controls, TWF diagnosis, 100 label controls, 20 paired audits |
| 10 | `experiments/10_review_assets.py` | revised manuscript tables, figures, and result macros; requires complete full validation |
| 11 | `experiments/11_cwru_download.py` | official source table and 36 checksummed recordings; downloads about 105 MB |
| 12 | `experiments/12_cwru.py` | nested grouped CWRU diagnosis and fixed-model acquisition-gain removal |
| 13 | `experiments/13_known_truth.py` | exact known-function controls; `--overlap` runs the separately planned overlap follow-up |
| 14 | `experiments/14_scale_replication.py` | optional, expensive higher-resolution reference extension; unfinished draws are not paper results |
| 15 | `experiments/15_temporal.py` | past-only context, nested chronological splits, paired cycle-bootstrap uncertainty |
| 16 | `experiments/16_effect_replication.py` | 40 observed audits, five generation seeds per size/condition; no new p-values |
| 17 | `experiments/17_scientific_assets.py` | complete-result-only scientific-extension tables and vector figure |
| 18 | `experiments/18_cwru_followup.py` | all 16 CWRU feature-view/split conditions; run before stage 17 |
| 19 | `experiments/19_verify_science.py` | historical hashes, scientific counts, and PDF bindings |
| 20 | `experiments/20_learned_calibration.py` | 7,200 learned finite-world functions; pooled calibration and transfer stresses |
| 21 | `experiments/21_fixed_learner.py` | 200 fresh calibration batches per eligible fitted function; all raw count vectors saved |
| 22 | `experiments/22_verify_learned.py` | fitted probability tables, exhaustive gates, exact scores, independent scalar checks |
| 23 | `experiments/23_learned_assets.py` | complete-result learned-calibration table and vector figure |

Audit stages read stage 1 parameters and saved results under `experiments/results/`; stages 4, 5,
and 10 generate manuscript assets. The reproduction script copies common assets into the current
template directory before building it. Seeds and package versions are fixed; compare regenerated
outputs with the saved artifacts rather than assuming numerical identity across environments.
Unit tests for the audit and validation helpers: `python -m pytest`.

## Layout

```
dataset/ai4i2020.csv      raw data
src/pdm_audit/            package: config, data, models, audit statistics, metrics, simulator
src/predict.py            CLI inference on new rows
experiments/              numbered pipeline stages; results/ holds their outputs
tests/                    unit tests
models/                   saved model, preprocessor, tuned hyperparameters
paper/ieee_bigdata/       earlier IEEE draft; common generated table bodies
paper/main.tex            earlier preprint, generated from the earlier IEEE draft
paper/figures/            generated figures
notebooks/                exploratory notebook
reproduce.sh              runs everything in order
requirements.txt          pinned versions
```

## October 22 review revision

The revised submission source is `IEEE-conference-template-062824/IEEE-conference-template-062824.tex`.
The files under `paper/` remain earlier manuscript variants. Build the revised paper with:

```bash
make -C IEEE-conference-template-062824
```

Additional review validation is defined in `experiments/REVIEW_VALIDATION_PROTOCOL.md`:

```bash
python experiments/09_review_validation.py --jobs 4
python experiments/10_review_assets.py
make -C IEEE-conference-template-062824
```

The validation stage writes a separate `experiments/results/review_validation.json` and checkpoint
directory; it does not overwrite the six historical result files. It includes 100 conditional
shuffled-label controls and 20 paired generated-data audits, each with its own 200-draw reference.
This stage takes roughly an hour on the development machine; the full historical scaling pipeline
can take several more hours. `--quick` writes smoke-test outputs and does not support paper claims.
The asset stage refuses incomplete or quick results. Checkpoints permit recovery after interruption.

See `IEEE-conference-template-062824/REVIEW_RESPONSE.md` for the changes and unresolved scientific
limits. The revised paper studies reference design and diagnostic failures; it does not establish
a generally calibrated mechanism-correctness test or real-equipment validity.


## Equal-count operating-region confirmation

The separately specified confirmation compares 200 uniform-population and 200
region-matched reference batches on each of 1,200 fresh high-overlap tests.
All 480,000 additional reference batches are saved and independently verified.
False warnings fall from 22.5--28.5% to 4--7.5%; every matched binomial interval
includes 5%. This controls reference count, not data-acquisition cost, and assumes
known operating-region metadata rather than an estimated field label.

Matching also weakens correct context localization for forests (133 to 48 of
200; 136 to 62 of 200) and logistic regression (75 to 6 of 199; 75 to 17 of 195).
Tree localization is unchanged. Predictions are identical across references,
and context permutation still reduces F1. These are measured failure cases,
not an impossibility theorem or a universal detection tradeoff.

Scripts 24--25 retain and verify the earlier filtered-bank sensitivity study;
scripts 26--27 execute and verify the equal-count confirmation. Protocols are
`experiments/OPERATING_REGION_PROTOCOL.md` and
`experiments/REGION_BUDGET_PROTOCOL.md`. Script 23 generates the final manuscript
figure and table from the confirmation. No historical numerical result changes.


## Measured alignment and metadata follow-up

Completed October 23 under `experiments/INDUSTRIAL_ALIGNMENT_PROTOCOL.md`.
All 36 official table rows match load, location, diameter, fixed outer-race
position, and source URL; measured RPM differs by at most 9 RPM. Forty-two
binary forests cover every load/diameter holdout, location, and full/own-anchor
view, with nested group selection. All 21 restricted heads pass exhaustive
split-support checks and exact gain invariance. Their diameter-held-out pooled
recording F1 is 0.611/0.800/0.500 for ball/inner/outer faults. Restriction therefore
certifies functional feature support, not physical correctness or robust diagnosis.

For the paper's all-mechanism S batches, only three non-overlapping inventory
batches exist per load. Reserving a test leaves at most two matched calibration
batches; pooling leaves at most eleven, before training consumes any records.
Neither permits a finite 5% rank threshold; the honest outcome is abstention.
These counts are optimistic because bearing identity and exchangeability remain
unverified. Window resampling does not create additional independent acquisitions.

Scripts 28--30 fit, independently replay all saved models/interventions, and
export every result plus the main-paper table. The full evidence and acquisition
requirements are in `IEEE-conference-template-062824/INDUSTRIAL_VALIDATION.md`.
The metadata and functional-control checks are complete. Industrial causal
alignment and nominal 5% warning calibration remain unmet gates; no historical
numeric result was overwritten.


## Paderborn feasibility gate

The October 23 source audit is complete: 32 documented bearing IDs and four
operating regions, with secondary/mixed component labels retained. The 2,560
advertised capture names are expected metadata, not verified signals. Original
archive connection attempts failed from this environment; raw headers and logs
remain unverified. The registry flags a KA04/KI18 narrative inconsistency rather
than silently overriding source tables.

For the existing failure-only S batch target on new bearings, all 26 damaged
bearings yield at most 13 disjoint two-bearing batches before training/test
reservation, or 12 calibration batches after one test. A finite 5% rank threshold
requires at least 19. Pure IR/OR pairs give an even smaller bound. Repeated
captures and operating conditions do not increase bearing-cluster counts.
Physical alignment and acquisition exchangeability are separate unverified gates.

No model or calibration experiment ran. A bearing-disjoint future diagnosis
partition is frozen in `experiments/PADERBORN_SPLIT_CONTRACT.json`; it is not
presented as an executed or valid S-calibration study. See
`IEEE-conference-template-062824/PADERBORN_FEASIBILITY.md`. Run scripts 31 and 32
to regenerate and independently verify the audit from cached official sources.
The previously verified conference manuscript PDF remains unchanged.


## Raw physical/acquisition follow-up supersedes summary-only interpretations

Two training-bearing archives (KA04/KI04) were inspected through a checksum-
verified Zenodo mirror; byte identity against the unreachable university host
remains unverified. All 160 captures were checked. Three duplicate groups in
KA04 leave 156 distinct numerical payloads across both bearings; direct checks
match all sensor and relative-time arrays within each group. One replacement is
explicitly documented in the bundled measurement log. Distinct payloads do not
establish independent acquisitions or independent bearings.

KI04's detailed profile records IR pitting plus OR indentations, invalidating
its earlier summary-table interpretation as pure IR. The old registry remains
a historical table transcription, not complete physical truth. A separate
`experiments/PADERBORN_PHYSICAL_LABEL_OVERLAY.json` blocks using that pure-label
interpretation without an explicit revision; frozen role IDs remain unchanged.
No reserved-calibration/test signals were inspected and no model was trained.

The predeclared spectral screen passes both candidate locations in all 160
captures under both timing variants. It is nonspecific and does not verify
exhaustive physical alignment. Acquisition independence and bearing
exchangeability remain unverified; the 5% new-bearing calibration gate stays
closed. Scripts 33, 35, and 34 execute, export, and independently verify this
follow-up. Scripts 31/32 reproduce only the earlier documentation-level audit.
The main paper PDF is unchanged. Full evidence:
`IEEE-conference-template-062824/PADERBORN_PHYSICAL_ACQUISITION.md`.
