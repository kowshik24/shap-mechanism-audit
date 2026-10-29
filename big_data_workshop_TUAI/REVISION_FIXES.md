# TUAI manuscript revision checklist

All twelve items from the latest manuscript review have been addressed locally. The author list and contact details use the user's explicit confirmation. Existing experimental outputs were retained; experiments were not rerun.

1. **Introduction:** Replaced the absolute usefulness claim with the diagnostic distinction between expected conditions, artefacts, and downstream responses. Described AI4I as providing mechanism indicators rather than recording a mechanism for every failure.
2. **Predictive split and audit:** Distinguished the reserved predictive test split from attribution cross-fitting over all 10,000 rows. Clarified that retuning the null does not make the original-label audit nested.
3. **Inference:** Defined each verdict through Holm-adjusted p-values. Limited primary familywise-error control to each four-mechanism statistic family, conditional on valid underlying p-values. Listed tested K values explicitly.
4. **Grouped and temporal evaluation:** Specified four recording-group folds for CWRU and five run-group folds for TEP. Explained that within-run label permutations do not preserve temporal label dependence or establish exchangeability.
5. **Verdict matrix:** Reduced the main matrix from 34 to 20 variants, removed tiny p-value annotations, and increased the table text size. The supplement retains the full p-value table and now explicitly includes separate ADASYN and 10-neighbour Borderline-SMOTE results. Table generation supports `--workshop-only` to avoid writing unrelated paper tables.
6. **Robustness interpretation:** Replaced “loses its relevant share” with “no longer passes the relevant-share test.” Qualified the proposed explanation for the raw-feature ablation.
7. **Attribution comparisons:** Corrected the KernelSHAP sentence, limited the figure-caption comparison to HDF/PWF/OSF, scoped normal-operation references to tested variants, and distinguished LIME probability outputs from logit explanations.
8. **Computational cost:** Limited the 5(K+1) count to five-fold audits with fixed hyperparameters and acknowledged additional search fits.
9. **TEP and LSTM:** Removed the inference that flags on different feed valves establish a shared control response. Identified F1 as pooled fault-active-class F1 across both fault types. Added code-verified grouped evaluation, scaling, optimizer, threshold, attribution aggregation, and full null-refitting details to the supplement.
10. **Ablations:** Renamed the comparison as audit-component ablations on selected diagnostic checks. Distinguished the lack of a global-ranking statistical verdict from a failed test and avoided a general superiority claim.
11. **Ending:** Moved Discussion and Limitations before Conclusion. Shortened the conclusion to keep it together on page 8 and avoid repeated qualifications.
12. **Authors, artifacts, and presentation:** Inserted Koshik Debanath, Prakrit Uprety, and Peiyu Li with the supplied affiliation and emails. Replaced the mutable branch citation with a content-addressed computational snapshot. Moved redundant pseudocode into the supplement under its own heading, retained the visual overview, and inspected every page of both rebuilt PDFs.

## Verification

- Main manuscript: 10 pages. Supplement: 5 pages.
- All 80 verdicts in the 20 retained matrix rows match the previous matrix.
- Compilation: no LaTeX warnings, overfull boxes, undefined citations/references, or multiply defined labels.
- All 77 files in the computational snapshot match its internal SHA-256 manifest and the current working files. The archive checksum is independently verified.
- Final PDF checksums are recorded in `artifacts/validation.json`.
- No existing experiment result files were changed by this revision. Scientific validity was not re-established through a new experimental run.

## Artifact scope

`artifacts/snapshot.json` identifies the immutable local computational archive and its full checksum. It includes source code, saved JSON/CSV results, trained model artifacts, the AI4I CSV, and environment files. Raw CWRU and DTU recordings are not bundled. The archive is a computational snapshot, not a copy of the final manuscript. The main PDF and supplement accompany it separately.

Public hosting was not verified. The manuscript and supplement do not claim that this local snapshot has been published. No upload, push, or submission was performed.

## Rebuild

From the repository root, regenerate the workshop tables with:

```sh
.venv/bin/python experiments/04_tables.py --workshop-only
```

From `big_data_workshop_TUAI`, build into a temporary directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/tuai-fix-build IEEE-conference-template-062824.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/tuai-supp-fix-build supplement.tex
```

The deliverables in this directory were copied from these verified builds. Rebuilding the computational code or stored results requires a new snapshot and corresponding artifact citation; a Git branch name is not an immutable identifier.

## Third review round (paperreview.ai, 2026-10-29)

New experiments (results in `experiments/results/`, logs in `experiments/logs/`):

- **TEP rotated-label null** (`15_tep_audit.py --null shift`): each run's time-ordered labels rotated by a random offset, keeping the fault-active block contiguous. Static and 30-minute-feature models, K=200: every verdict unchanged. LSTM not rerun. Files: `tep_audit_shift.json`, `tep_audit_window10_shift.json`.
- **Three weak proxies** (`24_multi_proxy.py`, K=500): three independent noisy copies of the HDF flag. At sigma=0.35 they take 0.38 of HDF attribution together (max 0.14 each); HDF not aligned but not flagged. Flagged only at sigma=0.25. File: `multi_proxy_3.json`.

Text changes from stored results: seed/background stability (R_m range <=0.02, identical top-3 features), per-mechanism raw-feature gap, effect of applying the 0.05 cutoff as a rule, multiplicity citations (Fisher 2019, Black 2022, Krishna 2024), learned conditional Shapley in future work, and a reporting checklist in the supplement.

Abstract now mentions shortcuts spread over several sensors. Main manuscript still 10 pages; supplement 7 pages.

New computational snapshot `artifacts/tuai-computation-5b97414ab7ef.zip` (81 files, every file read from commit 7238db1, deterministic rebuild) replaces `c0ec38a0b7f2` in the supplement and `refs.bib`. It adds stage 24 and the three new result files. The old archive is left untracked in `artifacts/` and is no longer cited.
