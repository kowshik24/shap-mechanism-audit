# Composite-reference addition

October 25, 2026. Added after observing the fixed-reference transfer results,
before running the composite calibration. This addition is exploratory. It does
not alter the earlier protocol or results.

Within each learner and n stratum, use the 200 disjoint 310000-series mechanism-
only fits as independent draws from the specified training procedure conditioned
on the full-state eligibility gate. Pair each with the first target-base batch
of its corresponding saved case. This produces one score per calibration fit,
not hundreds of correlated scores from each fit. Verify all 200 are eligible.

Evaluate the 200 independently trained 410000-series mechanism-only fits on
their 200 untouched batches, together with their fixed unrestricted candidates
under all five C laws from experiment 43. No evaluated mechanism-only fit is in
the calibration fit bank. The null is now a new eligible fit from the same
specified training procedure and target sampling law, not every acceptable model
or another learner family. Under this joint model-and-batch exchangeability,
global conformal ranks give marginal control, and within-bin ranks give the
corresponding Mondrian target. The guarantee is not conditional on a fixed
candidate's fitted function or a realized calibration bank.

Use global, original fixed overlap bins, and source-only tertile/quintile edges
estimated from the first source-base batch per saved case. Each bin uses only
calibration scores in that bin, at alpha=.05, with at least 19 needed for a finite
cut. SHAP uses strict ranks; finite-difference scores use independent randomized
rank ties. No second residual calibration is applied in this composite experiment.
Report calibration counts, abstention, self/null-model warnings, and pre/post
shift harm-conditional localization. Keep the harm floor .05 under error=.50.

Bootstrap intervals resample held-out fitted-model cases using shared indices
across methods. They condition on the single realized calibration fit bank.
The rank guarantee and a Beta/order-statistic tolerance calculation describe
calibration-bank randomness separately. Do not call case-bootstrap precision
uncertainty over new calibration banks or claim across-family validity.
