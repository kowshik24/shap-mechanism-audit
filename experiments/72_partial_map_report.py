"""Export the identification proof, measured ranges and concrete provenance request."""
import pdm_audit
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';PAPER=ROOT/'IEEE-conference-template-062824'


def fmt(x):
    if x is None:return 'unavailable'
    return '['+', '.join('−∞' if a=='-Infinity' else f'{a:.3f}' for a in x)+']'


def main():
    r=json.loads((OUT/'partial_map.json').read_text());proof=json.loads((OUT/'partial_map_verification.json').read_text());assert proof['verified']
    publisher=json.loads((OUT/'publisher_protocol_replay.json').read_text());assert publisher['verified'] and not publisher['physical_commands_executed']
    runtime=json.loads((OUT/'publisher_runtime_check.json').read_text());assert runtime['verified'] and not runtime['physical_commands_executed']
    report=['# Audit identification with incomplete feature maps','',
        'This extension replaces an assumed complete physical map with explicit bounds over compatible maps. It provides conditional partial identification, not a completed causal map, new acquisition evidence, or a new conformal theorem. All original model fits and reported supplied-map scores remain unchanged.','',
        '## Map family and sharp ranges','',
        'For one mechanism, let P be features required by the declared correspondence, N independently excluded features, and U unresolved features. An admissible expected map E satisfies P ⊆ E ⊆ P ∪ U. The partition itself requires evidence; computation cannot validate it. Specify the audit relation first: causes of failure, downstream responses, and engineering inspection targets are different relations. Lack of a downstream response does not disqualify an upstream cause or another legitimate predictor. Let a and b be normalized own/comparison attribution shares, and d=a−b. Then the coordinatewise sharp identification ranges are','',
        r'\[R\in[\sum_{j\in P}a_j,\;1-\sum_{j\in N}a_j],\]',
        r'\[D\in\sum_{j\in P}d_j+[\sum_{j\in U}\min(d_j,0),\;\sum_{j\in U}\max(d_j,0)],\]',
        r'\[S\in[\max_{j\in N}d_j,\;\max_{j\in N\cup U}d_j].\]',
        '', 'Each endpoint is attained by an admissible map. Given shares, bounds cost O(d) per mechanism; enumeration is verification, not the algorithm. For relevance, include none/all of U. For specificity, include the negative/positive-difference unknown features. For outside excess, include all/none of U. The Cartesian product of these ranges is not asserted to be jointly attainable. The empty maximum is −∞, representing absence of an eligible outside-feature warning. Missing comparison cohorts make D/S unavailable.','',
        'These ranges quantify uncertainty about the expected-feature map conditional on fitted attributions. They are not confidence intervals for sampling variation, population prediction error, or physical causal effects.','',
        '## Conservative rank rule and proof','',
        'For a global audit score A=max_m S_m, maximize each endpoint over mechanisms to obtain L≤A≤U. Calibrate on upper endpoints U_i and warn only when the candidate lower endpoint exceeds their strict-rank cut. Let k=ceil((M+1)(1−alpha)); use +∞ if k>M. For any admissible actual map,','',
        r'\[\{L_{\mathrm{new}}>U_{(k)}\}\subseteq\{A_{\mathrm{new}}>A_{(k)}\}.\]',
        '', 'The inclusion follows from L_new≤A_new and A_(k)≤U_(k). Under the same exchangeable reference-score null as the ordinary rank rule, the marginal warning probability is therefore at most (M+1−k)/(M+1). This is a conservative order-statistic application; it does not establish the map family or physical exchangeability. Constraints must cover the actual map. The result is conditional on that validity and the frozen scoring pipeline. If a map estimate or reference procedure changes the score law, exchangeability must be justified for that procedure.','',
        'This removes the need to resolve every unknown feature before reporting relevance/specificity ranges. A robust outside warning still needs a physically justified exclusion with enough excess. If N is empty, a map containing every feature remains admissible and L=−∞; the causal-warning output is abstention, not a successful zero-false-alarm detector.','',
        '## Reconstructed study results','',
        f"Experiment 70 reconstructs {len(r['rows'])} score records: nine CWRU, six pooled SCA, twenty within-case SCA, and 680 DTU mechanism/packet records. No model is refitted. Required sets are the original partial correspondences; no causal exclusions are invented. Original relevance equals the new lower endpoint, original outside excess equals the upper endpoint, and original specificity lies inside its range.",
        '', '| CWRU condition | Head | Relevance range | Specificity range | Outside-excess range |','|---|---|---|---|---|']
    for row in [x for x in r['rows'] if x['study']=='CWRU']:
        label='Original' if row['condition']=='original' else ('Resonance' if row['view']=='matched' else 'Removed')
        report.append('| '+' | '.join([label,row['fault']]+[fmt(row['bounds'][key]) for key in ['R','D','S']])+' |')
    report.extend(['','For example, the original ball-head specificity point score is about 0.003, but its compatible-map range is [−0.082, 0.085]. Positive point specificity does not imply positive specificity across unresolved maps. This interval includes zero by construction when the all-feature map is admissible; that fact is algebraic, not a new measured hypothesis-test finding.','',
        'Within SCA cases the missing other-failure cohort remains missing: only relevance ranges are available. Pooled SCA remains a cross-fitted summary, not one fixed-function audit. For DTU, the calibration-upper cut exactly recovers the original 0.03137267468676029. The fixed-map mathematical control recovers the original 0/60 unaltered and 60/60 biased warnings. With physical exclusions unresolved, every physical-causal warning abstains; that is not evidence of 0% field error or absence of the injected reliance.','',
        '## Independent verification','',
        f"Experiment 71 independently rebuilds all {proof['identified_ranges']} share vectors from the original attributions and metadata. It enumerates {proof['exact_map_completions_enumerated']:,} compatible maps across the measured-bearing records and checks exact extrema, constructs endpoint witnesses for all 53-input process records, and checks rank-event inclusion over 120 permutations. Unit tests also enumerate every required/excluded/unknown partition for a four-feature case. The formulas, missing-cohort behavior, strict ranks, and no-exclusion abstention all pass.",
        '', '## Acquisition follow-up','',
        'The original publisher generator scripts were also executed locally to compile protocol text, without connecting to hardware. Their generated commands independently reproduce the same 32/57 source-record schedule matches and corroborate the 25 unresolved wind associations. Searching for a favorable alternative seed was not used to declare a match. A separate Python 3.11/NumPy 1.26.3/pandas 2.1.4 run produces identical text for all 28 wind protocols. This checked version difference does not explain the mismatch; the unpinned source requirements and unknown original runtime still require provenance records. The project remains on its pinned Python 3.13/package environment.','',
        'An unsent, concrete provenance request is provided in `PHYSICAL_PROVENANCE_REQUEST.md`. Independent physical units and effect exclusions cannot be supplied by interval calculations, filenames, numerical uniqueness or compiler replay. No outside party was contacted.','',
        '## Reproduction','',
        'With the verified source studies present, run `PYTHON=.venv/bin/python bash reproduce_partial_map.sh`. Outputs are `partial_map.json`, `partial_map_ranges.csv`, `partial_map_draws.npz`, `partial_map_verification.json`, and `publisher_protocol_replay.json` under `experiments/results/`. Settings and post hoc scope are in `experiments/PARTIAL_MAP_PROTOCOL.md`.'])
    report_path=PAPER/'PARTIAL_MAP_REPORT.md';report_path.write_text('\n'.join(report)+'\n')
    request='''# Acquisition provenance request — prepared, not sent

The purpose is to resolve source-to-acquisition associations and define defensible
physical grouping, not to allege invalid experiments. No email, issue, or other
external message has been sent.

## Verified evidence supplied with a request

We authenticated the author-linked `lt_validate_v1` and `wt_validate_v1` ZIPs
against the published MD5 values, retained every CSV/member checksum, and compiled
the unmodified published generators at dataset-repository commit
`0c8a77ea14bd4437b1fdd79adc1aa91ac2342569` with the pinned configurations.

All 29 light acquisitions and three wind acquisitions match the generated
treatment schedules. The other 25 wind acquisitions do not match those exact
schedules. `validate_load_out.csv` has 49 rows, while its configuration specifies
50. The complete per-file comparison, compiled protocol text and hashes are in
`experiments/results/publisher_protocol_replay.json`; acquisition checks are in
`experiments/results/chamber_acquisitions.json`.

## Source records needed

1. The original executed protocol text or immutable protocol identifier for each
   wind acquisition, with a mapping to the released CSV/member hash.
2. Generator/configuration commit, seed and RNG/runtime versions used for those
   acquisitions, and any retry, interruption, trimming or postprocessing records
   explaining the 49-row capture. A different actual schedule may be valid; it
   needs an authenticated association rather than a seed search.
3. Device/chamber serial identity and session/run boundaries, acquisition date
   and clock origin, reboot/remount/maintenance records, and any sharing of noise
   or controller state between releases/experiments. ZIP filesystem dates are not
   treated as authenticated acquisition timestamps.
4. Reset, warm-up and stabilization logs or a documented dependence model. Which
   physical groups may reasonably be treated as draws from the same reference
   population, and under what operating conditions?
5. Define the audit relation (upstream causes, downstream responses or inspection
   targets). For any proposed exclusion, provide its physical/topological
   justification or intervention-based equivalence evidence, the feature units,
   effect margin, intervention levels and operating range. We recognize that the
   publication explicitly does not interpret a graph non-edge as proof of no
   causal effect, and do not request an unsupported universal zero-effect claim.

## What would resolve the research gate

Authenticated protocol/session grouping can repair the acquisition ledger.
It does not automatically establish independence or exchangeability. Physically
justified exclusions within a declared feature/context domain can narrow the
admissible-map family; unknown relationships remain unknown. Protected healthy/
faulty comparison cohorts and calibration/evaluation units are then needed before
claiming an industrial mechanism-warning operating characteristic. Physical
chamber interventions alone are not bearing-failure field validation.
'''
    request_path=PAPER/'PHYSICAL_PROVENANCE_REQUEST.md';request_path.write_text(request)
    files=[Path(__file__),report_path,request_path,OUT/'partial_map_verification.json',OUT/'publisher_protocol_replay.json',OUT/'publisher_runtime_check.json']
    (OUT/'partial_map_assets.json').write_text(json.dumps({'verified_inputs':True,'sha256':{
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
    print('Exported partial-map proof/results and unsent acquisition-provenance request')


if __name__=='__main__':main()
