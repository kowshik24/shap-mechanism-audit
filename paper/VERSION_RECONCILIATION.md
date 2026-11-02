> **Superseded.** This note describes an earlier 8-page draft. The paper revised on 2026-10-31 is `paper/main.tex` and `paper/main.pdf` (9 pages).

# Reconciliation with the supplied earlier manuscript

The supplied manuscript was used as a narrative reference, then reconciled with the current stored experiments. The active sources are `paper/main.tex` and `paper/supplement.tex`. This pass does not restore the earlier template authors, the description of Tennessee Eastman as future work, or claims superseded by later experiments.

## Section changes

- **Abstract:** A 187-word research narrative now moves from the industrial task and interpretability need to the mechanism-evaluation gap, the proposed audit, collective findings, and bounded implications. It defines SHAP and describes held-out observations in plain language. Individual model scores, fault lists, permutation counts, and experimental thresholds are omitted.
- **Introduction:** The background-to-problem progression from the supplied version is retained. The gap distinguishes model faithfulness, agreement between explainers, and agreement with failure mechanisms. The contribution is the integration of mechanism-level aggregation, held-out explanations, and a retrained statistical reference. The evaluation scope includes laboratory bearing signals, simulated process runs, alternative models and references, incomplete maps, and distributed shortcuts.
- **Related work:** The maintenance context and closest AI4I precedent are explicit. The discussion distinguishes mechanism-specific testing from global SHAP-guided feature selection, faithfulness benchmarks, and randomization checks. Attribution-reference dependence and model/explainer disagreement motivate the expanded experiments without asserting that any one explanation is ground truth.
- **Audit framework and evaluation design:** Core statistic and p-value equations are preserved. The main paper explains the audit target, decision rule, reference, study logic, and data-specific grouping. Detailed training settings and supporting diagnostics remain in the supplement. Non-nested selection and temporal randomization limits remain explicit.
- **Results and analysis:** Findings are connected to the questions they answer. The revised prose separates positive mechanism alignment, insufficient alignment, outside-set localization, and predictive performance. Numerical tables and figure assets are unchanged.
- **Discussion, conclusion, and future work:** The conclusion emphasizes what the experiments collectively reveal about explanation reliability and what the diagnostic framework contributes. Limitations cover finite model coverage, incomplete maps, weak/distributed reliance, conditional intervals, exploratory effect thresholds, and absence of operational industrial validation. Future work targets prospective validation and aggregated outside-set tests; Tennessee Eastman is already evaluated.

## Earlier interpretations reconciled with current evidence

| Earlier emphasis | Current evidence and interpretation |
| --- | --- |
| Tool-wear disagreement may arise from correlated engineered features. | Background/input and grouped-attribution checks support attribution-reference dependence; they do not establish its sole cause. Independently recomputed wear/overstrain-product Spearman correlation is 0.92 overall and 0.13 within TWF observations. See `twf_diagnostics.json` and `dependence.json`. |
| Evaluation covers only two tree families and two SHAP formulations. | Current checks also include MLP attribution methods, temporal-feature XGBoost, and LSTM/Integrated Gradients. Coverage remains finite; no model-independent validity claim is made. |
| One fabricated sensor characterizes shortcut detection. | The three-proxy experiment shows a localization limit: at one setting, proxies receive 38% total attribution and at most 14% each. HDF loses alignment, but no proxy is flagged. This is not an aligned verdict. See `multi_proxy_3.json`. |
| Tennessee Eastman is a future dataset. | Static, temporal-feature, and LSTM results are included. Pooled fault-active-class F1 rounds to 0.76, 0.94, and 0.91. Reactor cooling aligns while condenser cooling is flagged. The flags need investigation; they are not established shortcuts. |
| A temporal null check applies to all process variants. | Circular rotations preserve the static and temporal-feature verdicts. The LSTM was not rerun under this null. Neither tested null establishes temporal exchangeability. See `tep_audit_shift.json` and `tep_audit_window10_shift.json`. |
| Only simulated maintenance data support the audit. | CWRU adds measured signals from seeded laboratory faults. Inner/outer race align under the stated bands; ball-fault conclusions depend on bandwidth. An injected acquisition gain preserves recording accuracy and flags outer-race reliance on log-RMS. These are not operating-plant data or proof of causal recovery. |
| More data establish reliable shortcut detection. | Strong single-sensor shortcuts are flagged in all ten generated-data runs, but weak reliance remains difficult to detect. The 300,000-observation condition has only one generation seed. Small TWF outside-set flags and a post-hoc 0.05 cutoff do not establish an operational decision threshold. |

Result files above are under `experiments/results/`; the correlation is recalculated from `dataset/ai4i2020.csv`. `paper/artifacts/experiment_claim_checks.json` records 28 passed checks against stored evidence. No model experiments were rerun in this reconciliation.

## Preservation and delivery checks

- The confirmed three-author block and the title's reference to **known** failure mechanisms are retained.
- All five core equation blocks and all 78 baseline evidence/table/figure files are unchanged.
- Both documents compile successfully, with no overfull boxes or unresolved citations/references. Final PDFs contain 8 manuscript pages and 9 supplement pages. Every rendered page was inspected, including pages affected by final edits; eight manuscript underfull-spacing diagnostics were reviewed visually.
- The local website abstract and PDF copies are synchronized. The computational snapshot is preserved and is not represented as a snapshot of this latest prose revision.
- Current PDF checksums and validation details are recorded in `paper/artifacts/validation.json`. Its top-level rerun history describes earlier computational work; this reconciliation has a separate entry.
- No commit, push, deployment, conference submission, or external message was performed.
