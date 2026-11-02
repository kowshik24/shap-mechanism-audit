> **Superseded.** This note describes an earlier 8-page draft. The paper revised on 2026-10-31 is `paper/main.tex` and `paper/main.pdf` (9 pages).

# Editorial revision: research narrative and reviewer-comment coverage

This record describes the preceding editorial pass. The subsequent reconciliation with the supplied earlier manuscript is documented in `VERSION_RECONCILIATION.md`; its updated abstract has 187 words. Historical counts below describe this preceding pass.

The current manuscript is `paper/main.tex`; the earlier workshop directory was consolidated into `paper/` before this revision. The original experimental results, numerical tables, and figure assets have not changed.

## Comment-by-comment response

1. **Abstract: motivation, gap, contribution, and overarching findings.** Rewritten as a 180-word narrative, reduced from 235 words. It first introduces predictive maintenance and the need for failure-related evidence, explains what attribution methods describe, states the evaluation gap, introduces the audit, and ends with collective findings and bounded implications. Individual classifiers, scores, fault names, p-values, statistic inventories, and experiment-by-experiment findings were removed.
2. **Do not jump to explanations too quickly.** The abstract now begins with the maintenance task and the diagnostic need. Feature attribution is explained before SHAP is introduced. SHAP is expanded on first use.
3. **Explain “held-out SHAP.”** The abstract uses “observations excluded from model training.” The introduction also explains that each observation is explained by a model fitted without it; the framework then introduces cross-fitting in context.
4. **Introduction: strengthen motivation and the evaluation gap.** The revised introduction distinguishes predictive success, faithfulness to a model, and agreement with failure mechanisms. The acquisition-artefact example motivates the problem. The proposed integration of mechanism-level aggregation, held-out explanations, and a retrained reference is stated as the contribution, without claiming a new attribution algorithm or universal superiority.
5. **Related work: clarify the framework's position.** Reorganized into predictive-maintenance explanations, what explanation evaluations establish, and reference dependence/disagreement. It distinguishes global feature-selection inference from mechanism-specific attribution evaluation and removes the catalogue of computational speed-up algorithms unrelated to the audit's contribution. Existing primary references were retained where they support the revised claims.
6. **Conclusion: collective meaning rather than a results inventory.** Rewritten around what the study reveals about explanation reliability, what the audit adds between prediction and physical interpretation, the limits of its sensitivity and mechanism maps, and the evidence needed before operational use. No individual experimental scores are listed. The conclusion is kept together rather than split across pages.
7. **Methods and experiments: reduce overload and refine logic.** The main paper now separates the Audit Framework (target, statistics, held-out reference) from Evaluation Design (data/expected sets and study logic). Results and Analysis form one sequence: primary alignment, why controls matter, stability/reference dependence, shortcut limits, and laboratory/process evidence. Each subsection connects evidence to its interpretation. Detailed feature equations, model searches, sampling schedules, normalization checks, supporting diagnostics, sweep thresholds, timings, and two supporting displays were transferred to the supplement.
8. **Keep interpretation bounded.** Non-nested selection, conditional bootstrap uncertainty, the temporal randomization caveat, weak/distributed shortcut failures, incomplete maps, exploratory effect thresholds, and absence of operating-plant validation remain explicit. The supplement's checklist no longer presents the post-hoc 0.05 excess cutoff as a validated default.

## Preservation and verification

- Main paper: 8 pages, reduced from 10. Supplement: 9 pages, previously 7; its growth preserves details removed from the main narrative.
- Six figures and five tables remain in the main paper. The explainer comparison and component-ablation display are retained in the supplement.
- All five statistic/p-value equations are textually unchanged; generated numerical tables and figure assets are unchanged.
- All 55 stored result/evidence files have unchanged SHA-256 hashes. No experiments were rerun for this editorial revision.
- Both documents compile without LaTeX warnings, overfull boxes, undefined references/citations, or duplicate labels. Every rendered page was inspected.
- The existing computational archive is unchanged; its archive checksum and all internal file hashes were verified. It remains a computational snapshot, not a frozen copy of this revised manuscript.
- Local website abstract and downloadable PDF copies are synchronized. No website deployment, push, submission, or external message was performed.
- Final PDF hashes and editorial checks are stored in `paper/artifacts/validation.json`. The existing top-level experiment-rerun entries in that file describe earlier work; this pass is recorded separately with `experiments_rerun: false`.

## Rebuild

From `paper/`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/tuai-editorial-build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/tuai-editorial-supp-build supplement.tex
```

The final PDFs were copied from these verified builds. Website copies are in `docs/static/pdfs/`; the abstract is mirrored in `docs/index.html`.
