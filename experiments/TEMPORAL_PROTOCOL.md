# Chronological wear-cycle evaluation protocol

Written October 23, 2026 before fitting the temporal extension. Earlier shuffled
split, CWRU, and oracle-control results are known. This is an internal prospective
analysis plan, not external preregistration.

Use the released AI4I sequence plus ten generated 30,000-row sequences, generation
seeds 20--29. Infer cycle boundaries where current tool wear is lower than the
previous row's wear. In generated data this tracks tool resets; in the release
these are provisional wear-reset groups, not documented physical tool IDs or
timestamps. Never use a future reset to construct a predictive feature.

Assign whole cycles chronologically: first 60% training, next 20% validation,
last 20% test. Within training, use expanding chronological folds: train first
30%, validate next 20%; train first 50%, validate next 20%; train first 70%,
validate last 30%, always in whole cycles. Choose max_depth from [3,6,8], using
mean inner-fold failure F1 at threshold 0.5, ties favoring smaller depth. Other
XGBoost settings and within-training Borderline-SMOTE match the historical
configuration. Training/model seed is 42. The untouched outer validation cycles
select the final F1 threshold; test cycles do not select parameters or threshold.

Compare the nine existing numerical features against those features plus:
previous wear, previous torque, previous temperature difference, elapsed sample
count since the latest observed reset, and mean torque of up to five preceding
rows within that cycle. No previous mechanism flags or failure labels are inputs.
At cycle start use previous wear 0, current torque/temperature for missing lag
measurements, and current torque for empty history. These conventions use only
information available at the current observation. The generated temperature
process remains iid; this is not a faithful time-series industrial simulator.

Report overall failure F1 and per-mechanism recall, including TWF counts, on the
same test cycles. Summarize paired differences across ten generation seeds.
Within each sequence use a paired cluster bootstrap of test cycles (3,000 draws,
seed 20261023), retaining all rows from each sampled cycle and the same draw for
both models. Omit undefined recall draws and report their count. Do not treat
rows or overlapping flags as independent replications. The intervals condition
on the fitted models and observed cycle inventory; they do not include the full
parameter-selection uncertainty. Preserve non-improvements and negative results.

This study tests chronological performance and a small, fixed set of past-only
features. It does not claim that temporal dependence is fully solved, that
stochastic TWF is deterministic, or that a particular feature ranking is causal.
