"""
Stage 22: the TEP audit with a sequence model.

Same runs, samples, labels, mechanisms, E_m and within-run null as stage 15, but each sample is the
sequence of the last W samples (3 min apart) of all 53 variables, classified by a one-layer LSTM
(32 units) on per-variable standardized inputs. Attributions are Integrated Gradients of the logit
from the all-zero (training-mean) sequence; a variable's attribution is the absolute sum of its
signed attributions over the W time steps, so the statistics stay on the 53 variables.

Usage: python experiments/22_tep_lstm.py [--quick] [--window 10] [--K 200]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from pdm_audit.audit import holm, permutation_p
from pdm_audit.config import RESULTS_DIR, ROOT, SEED

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--window', type=int, default=10)
ap.add_argument('--K', type=int, default=200)
args = ap.parse_args()
K, W = (4 if args.quick else args.K), args.window
EPOCHS, STEPS = (3, 16) if args.quick else (20, 64)

EXPECTED = {'IDV4': [8, 20, 50], 'IDV5': [10, 21, 51]}  # stage 15's loops, as column indices of the 53
MECH = list(EXPECTED)

seqs, meta = [], []
for fault in (4, 5):
    for run in range(1, 201):
        full = np.fromfile(ROOT / 'data' / 'dtu_tep' / 'prospective_runs' / f'{fault}_{run:03d}.bin', dtype='<f8')
        full = full.reshape(2001, 54)
        idx = np.abs(full[:, 0, None] - np.r_[np.linspace(10, 25, 16), np.linspace(40, 95, 16)]).argmin(axis=0)
        for i in idx:
            seqs.append(full[i - W + 1:i + 1, 1:])
            meta.append((fault, run, full[i, 0]))
S = np.stack(seqs).astype(np.float32)  # (n, W, 53)
meta = pd.DataFrame(meta, columns=['fault', 'run', 'hour'])
y = (meta['hour'] > 30).astype(int).to_numpy()
groups = (meta['fault'] * 1000 + meta['run']).to_numpy()
fault_of_row = meta['fault'].to_numpy()
mech = {f'IDV{f}': ((fault_of_row == f) & (y == 1)) for f in (4, 5)}


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(53, 32, batch_first=True)
        self.head = torch.nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def fit(X, labels, seed):
    torch.manual_seed(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt, yt = torch.tensor(X), torch.tensor(labels, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        for b in torch.randperm(len(Xt), generator=g).split(256):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]), yt[b]).backward()
            opt.step()
    return net.eval()


def integrated_gradients(net, X):
    """IG of the logit from the zero sequence (the standardized training mean), trapezoid rule."""
    x = torch.tensor(X)
    grads = torch.zeros_like(x)
    for k, a in enumerate(torch.linspace(0, 1, STEPS + 1)):
        xi = (a * x).requires_grad_(True)
        net(xi).sum().backward()
        grads += xi.grad * (0.5 if k in (0, STEPS) else 1.0) / STEPS
    return (x * grads).detach().numpy()


def cross_fit(labels, seed=SEED):
    """Held-out |per-variable attribution| of the fault-active samples (others zero) and probabilities."""
    torch.set_num_threads(1)
    A, proba = np.zeros((len(S), 53)), np.zeros(len(S))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(S, fault_of_row, groups):
        mu, sd = S[tr].reshape(-1, 53).mean(axis=0), S[tr].reshape(-1, 53).std(axis=0) + 1e-6
        Z = ((S - mu) / sd).astype(np.float32)
        net = fit(Z[tr], labels[tr], seed)
        with torch.no_grad():
            proba[te] = torch.sigmoid(net(torch.tensor(Z[te]))).numpy()
        te_f = te[y[te] == 1]
        A[te_f] = np.abs(integrated_gradients(net, Z[te_f]).sum(axis=1))
    return A, proba


def statistics(A):
    out = {}
    for m in MECH:
        own, rest = A[mech[m]].mean(axis=0), A[(y == 1) & ~mech[m]].mean(axis=0)
        a, b = own / own.sum(), rest / rest.sum()
        excess = np.delete(a - b, EXPECTED[m])
        others = np.delete(np.arange(53), EXPECTED[m])
        out[m] = {'R': float(a[EXPECTED[m]].sum()), 'D': float(a[EXPECTED[m]].sum() - b[EXPECTED[m]].sum()),
                  'S': float(excess.max()), 'S_feature': int(others[excess.argmax()]), 'top_feature': int(a.argmax())}
    return out


def null_draw(k):
    rng = np.random.default_rng(SEED * 100_003 + k)  # same within-run permutations as stage 15
    perm = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        perm[idx] = rng.permutation(y[idx])
    return statistics(cross_fit(perm)[0])


t0 = time.time()
A, proba = cross_fit(y)
obs = statistics(A)
pred = (proba >= 0.5).astype(int)
tp, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
draws = Parallel(n_jobs=-1)(delayed(null_draw)(k) for k in range(K))
p = {f'{m}.{s}': permutation_p(obs[m][s], [d[m][s] for d in draws]) for m in MECH for s in 'RDS'}
results = {'K': K, 'window': W, 'observed': obs, 'f1': float(2 * tp / (2 * tp + fp + fn)),
           'recall_by_mechanism': {m: float(pred[mech[m]].mean()) for m in MECH},
           **{f'p_holm_{s}': holm({m: p[f'{m}.{s}'] for m in MECH}) for s in 'RDS'},
           'null_mean': {f'{m}.{s}': float(np.mean([d[m][s] for d in draws])) for m in MECH for s in 'RDS'},
           'seconds': time.time() - t0}
print(f'F1={results["f1"]:.3f} recall={results["recall_by_mechanism"]} ({results["seconds"]:.0f}s)')
for m in MECH:
    print(f'  {m}: R={obs[m]["R"]:.3f} (null {results["null_mean"][m + ".R"]:.3f}, p={results["p_holm_R"][m]:.3f})  '
          f'D={obs[m]["D"]:.3f} (p={results["p_holm_D"][m]:.3f})  C={obs[m]["S"]:.3f} var{obs[m]["S_feature"]} '
          f'(p={results["p_holm_S"][m]:.3f})  top=var{obs[m]["top_feature"]}')
name = 'tep_lstm_quick.json' if args.quick else 'tep_lstm.json'
(RESULTS_DIR / name).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / name)
