"""Publish the finite-grid physical-evidence assessment and remaining requirements."""
import pdm_audit
import hashlib,json,math
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';PAPER=ROOT/'IEEE-conference-template-062824'


def main():
    r=json.loads((OUT/'chamber_physical_map.json').read_text());proof=json.loads((OUT/'chamber_verification.json').read_text());assert proof['verified']
    source=json.loads((OUT/'chamber_sources.json').read_text());ledger=json.loads((OUT/'chamber_acquisitions.json').read_text())
    basis=Counter(p['support_basis'] for p in r['pairs'])
    report=['# Physical acquisitions and causal feature maps: verified evidence','',
        'This study authenticates real physical intervention recordings and independently reconstructs their protocols and response-map arithmetic. It does not certify independence between physical machines or an exhaustive causal truth map. The distinction is an observed result, not a substituted simulation claim.','',
        '## Primary sources','',
        '- Gamella, Peters and Bühlmann, [Causal chambers as a real-world physical testbed for AI methodology](https://www.nature.com/articles/s42256-024-00964-x), Nature Machine Intelligence 7, 107–118 (2025).',
        f"- [Pinned dataset repository](https://github.com/juangamella/causal-chamber/tree/{source['data_commit']}): `lt_validate_v1` and `wt_validate_v1`.",
        f"- [Pinned graph source](https://github.com/juangamella/causal-chamber-package/tree/{source['graph_commit']}/causalchamber/ground_truth).",
        '- Frozen local protocol: `experiments/PHYSICAL_INTERVENTION_PROTOCOL.md`. Source-replay findings and failed-acquisition handling: `experiments/PHYSICAL_INTERVENTION_RUN_NOTE.md`.',
        '', '## What was authenticated','',
        f"Both complete ZIPs ({sum(x['bytes'] for x in source['datasets']):,} bytes total) match the MD5 values in the pinned author READMEs. Every member also passes ZIP CRC and local SHA-256 verification. These are light/wind tunnel measurements, not simulated values or bearing-failure data.",
        '', '| Dataset | Acquisition files | Measurement rows | Exact protocol replays | Assessed channel pairs | Response-supported pairs |',
        '|---|---:|---:|---:|---:|---:|']
    for x in r['summary']:
        report.append(f"| {x['dataset']} | {x['source_configurations']} | {x['rows']:,} | {x['passed_acquisitions']} | {x['assessed_pairs']} | {x['response_supported_pairs']} |")
    report.extend(['','Replay checks row counts, treatment flags from the published RNG, intervention values, fixed controls, monotone local clocks, counters, finite signals, and numerical duplicates. All light files and three wind files pass. Twenty-five wind files do not replay the pinned treatment schedule; one of them, `validate_load_out.csv`, contains 49 rather than the prescribed 50 rows. All raw files and discrepancies are retained. A replay mismatch is not proof that the source experiment is invalid; it prevents this audit from claiming reconstruction of its exact randomization protocol.','',
        'No complete numerical acquisition payload is duplicated. The CSVs do not supply physical device serials, independently authenticated session IDs, or enough clock anchoring to verify independent physical sessions. ZIP filesystem timestamps are retained but are not authenticated acquisition dates. Setup waiting times and unique counters are not reset or independence certificates. A distinct file is an acquisition sequence, not automatically an exchangeable calibration unit. Different interventions/settings are also different data laws. The source RNG uses the same fixed seed in both dataset generators; this documents the intended schedule rather than proving independent physical random streams.','',
        '## Every-pair feature-map assessment','',
        f"The fixed universe contains eleven recorded measurement channels per chamber, including the light tunnel's two voltage diagnostics. All 57 protocols × 11 channels = 627 pairs are retained. Two response endpoints per pair give one prespecified Holm family of {r['hypothesis_family_size']:,} tests. Failed-replay acquisitions receive family-input p=1, while their descriptive counts remain available.",
        '', '| Evidence state | Pairs | Meaning |','|---|---:|---|',
        f"| Response and published graph | {basis['response_and_graph']} | Locally replayed randomized response plus a published directed path |",
        f"| Response only | {basis['response_only']} | Replayed response in a channel omitted from the graph |",
        f"| Published graph only | {basis['graph_only']} | Source path, without local adjusted response confirmation |",
        f"| Unknown | {basis['unknown']} | No supported response/path in the retained assessment |",
        '| Causally excluded | 0 | No physical zero-effect or practical-equivalence certificate |','',
        'The response screen uses treatment versus above-pooled-median and above-median-deviation indicators. Two-sided Fisher tests condition on assignment counts. Their randomization interpretation assumes the stipulated fair-assignment design and the global sharp no-effect null; it does not require independent sensor noise under that null. Rejection establishes a protocol-specific distributional response. Carryover under alternatives, thresholds that miss other distributional changes, and the narrow exposure settings prevent a universal or instantaneous structural interpretation. Numerically zero raw p-values reflect floating-point underflow, not zero null probability.','',
        'For example, changing the light current-sensor range `v_c` changes the recorded regulator-voltage diagnostic `v_reg`, a channel omitted from the source adjacency matrix. Its raw medians are 1023 and 824 under the two settings; the adjusted dispersion-endpoint p-value is approximately 1.14e-11. This is a measured response, not a newly established direct physical edge or a contradiction of the authors\' explicitly non-exhaustive graph.','',
        'All 57 intervention maps retain unresolved channels. The complete finite-grid ledger is therefore not an exhaustive causal map. A non-edge or non-rejection never becomes a negatively excluded feature. The published paper itself allows unvalidated effects and unmeasured confounding.','',
        '## Consequence for the SHAP audit','',
        'Keep three evidence states: supported features E+, excluded features E−, and unresolved features E?. These sets partition the declared feature universe, with each state carrying source/protocol/effect-scope evidence. The existing S score remains attribution outside a supplied expected set. It is not a causal-shortcut test when that outside set includes E?. A physical exclusion warning would maximize only over E−; an empty E− or unsupported acquisition population requires abstention from that claim. Descriptive attribution may still be reported.','',
        'This public study strengthens the physical evidence over an input-rule simulator, but it does not justify a new-machine 5% warning guarantee. No additional model is fitted to convert these source limitations into an apparent validation result.','',
        '## Reproduction and independent verification','',
        'Run `PYTHON=.venv/bin/python bash reproduce_physical_map.sh`. Experiment 66 downloads/authenticates source records; 67 reconstructs protocols and the entire map; 68 independently replays archive checks, flags, all 2×2 tables, exact hypergeometric probabilities, Holm adjustment, coverage and failed-source abstentions; 69 exports this report. The machine-readable outputs are `experiments/results/chamber_acquisitions.json`, `chamber_physical_map.csv`, `chamber_physical_map.json`, and `chamber_verification.json`.'])
    (PAPER/'PHYSICAL_CAUSAL_MAP_REPORT.md').write_text('\n'.join(report)+'\n')
    mcal=math.ceil(math.log(.05)/math.log(.95));ntest=math.ceil(math.log(.025)/math.log(.95));pairs=627
    plan=['# Acquisition and causal-map completion design','',
        'This is a prospective design, not completed evidence or a claim that a universal exhaustive physical graph can be proved from finite data. No physical system was actuated and no outside party was contacted.','',
        '## 1. Define the physical population and units','',
        'Specify equipment, physical device/bearing identity, acquisition/session identity, operating region, maintenance/remount/reset events, calibration history, and fault/inspection provenance. Retain source records with UTC clock anchoring and signed or otherwise authenticated acquisition logs. Group every dependent measurement from one physical unit/session before splitting. Disjoint IDs or filenames alone are insufficient. Randomize intervention order and record reset/stabilization evidence; repeated measurements without justified resets are dependent trajectories.','',
        '## 2. Define the map and what “excluded” means','',
        'First state whether the map concerns upstream causes, downstream responses, or inspection targets. Lack of a downstream response does not exclude a legitimate upstream predictor. Freeze all candidate features, their raw-sensor lineage, intervention levels, measured operating contexts, and an application-specific effect margin. Maintain supported/excluded/unknown states. A practical exclusion can mean that the intervention changes a channel\'s distribution by at most delta in a stated distance throughout the declared context grid, with simultaneous confidence. It is not an absolute assertion of no causal effect at every possible setting. Engineering topology/equations and independently reviewed inspection evidence may support additional exclusions, but these must be recorded as assumptions with their operating range.','',
        '## 3. Collect evidence for exclusions, not just non-significance','',
        'Use genuinely independent reset trials or an explicitly justified clustered/time-series design. Separate map construction from downstream calibration/testing. Include positive controls, zero-path controls, healthy controls and intervention-compliance measurements. Use equivalence confidence bounds for the complete prespecified effect family; failure of an ordinary significance test is not exclusion. Never apply an i.i.d. bound to arbitrary timestamp rows.','',
        f"For illustration, with independent trials and Kolmogorov distribution distance, simultaneous two-sample DKW bounds over L={pairs} pairs at total error 0.05 give radius sqrt(log(4L/0.05)/(2n_A)) + sqrt(log(4L/0.05)/(2n_B)). Even if empirical distance is zero, certifying delta=0.1 needs at least {math.ceil(2*math.log(4*pairs/.05)/.1**2):,} independent trials per treatment arm. This is a conservative sufficient design example, not a required universal sample count or a bound applicable to the current sensor rows. Observed nonzero distances need more data, or remain unresolved.",
        '', '## 4. Reserve calibration and evaluation acquisitions','',
        'Freeze one physically justified reference fitting recipe, the grouping law, comparison cohorts, backgrounds, and error target before reading calibration/evaluation outcomes. If randomness over fitted models belongs to the null, use separate seeded fits paired with distinct acquisition groups. Conditioning on a common frozen training corpus is allowed only when stated explicitly. Every packet must contain the own/other-failure cohorts needed for R/D/S; a file lacking them is not a complete calibration batch.','',
        f"Nineteen units give finite 5% strict-rank resolution but weak conditional precision. A more conservative global maximum-score cut with {mcal} i.i.d. calibration scores from the specified null has a 95% order-statistic tolerance upper bound below 5% (under a continuous score law; ties are conservative). Exchangeability alone gives the marginal rank guarantee, not this i.i.d. tolerance statement. Evaluating zero warnings in {ntest} independent units gives a two-sided 95% exact-binomial upper endpoint below 5%, conditional on that realized bank and stated replicate assumptions. These are different uncertainty statements, and observing any warnings changes the latter calculation.",
        '', 'For a deliberately simple three-mechanism packet assembled from three distinct damaged bearings plus one healthy control, those 59+72 packets would require 524 distinct physical bearing groups after reserving training data. This is an example of a sufficient packet design, not a claim that all valid audits require 524 bearings or 524 machines. Alternative designs must justify their own independent units and cohort support.','',
        '## 5. Completion gate','',
        'A causal exclusion warning requires independently supported acquisition grouping/exchangeability, verified labels, a declared finite feature/context universe, adequate comparison cohorts, a nonempty excluded set with simultaneous effect-scope evidence, and protected calibration/evaluation units. Unknown relationships remain unknown. The current physical dataset authenticates interventions and exposes map gaps but does not pass that completion gate.']
    (PAPER/'PHYSICAL_ACQUISITION_PLAN.md').write_text('\n'.join(plan)+'\n')
    files=[Path(__file__),OUT/'chamber_verification.json',PAPER/'PHYSICAL_CAUSAL_MAP_REPORT.md',PAPER/'PHYSICAL_ACQUISITION_PLAN.md']
    (OUT/'chamber_assets.json').write_text(json.dumps({'verified_inputs':True,'sha256':{
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
    print('Exported physical evidence report and prospective completion design')


if __name__=='__main__':main()
