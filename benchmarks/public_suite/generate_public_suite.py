# ruff: noqa
"""Rebuild the tracked Linear Lens public evidence suite deterministically."""
from __future__ import annotations
import csv, hashlib, json, math, os, random, time
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_linnerud, load_wine, make_classification, make_regression, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'benchmarks/public_suite/datasets'
MODELS = ROOT / 'benchmarks/public_suite/models'
RESULTS = ROOT / 'benchmarks/public_suite/results'
for p in (DATA, MODELS, RESULTS):
    p.mkdir(parents=True, exist_ok=True)
SEED0 = 20260813
ZCRIT = NormalDist().inv_cdf(0.9)
EPS = 1e-12

def influence(X, W):
    mu = np.mean(np.abs(X[:, None, :] * W[None, :, :]), axis=0)
    denom = np.sum(mu, axis=1, keepdims=True)
    return np.divide(mu, np.maximum(denom, EPS), out=np.zeros_like(mu), where=np.ones_like(mu, dtype=bool))

def entropy(P):
    return -np.sum(P * np.log(P + EPS), axis=1)

def zscore(v):
    sd = float(np.std(v))
    if sd < EPS:
        return np.zeros_like(v)
    return (v - np.mean(v)) / sd

def roles(z, deep=False):
    out = []
    for q in z:
        if q < -ZCRIT:
            out.append('unimodal' if deep else 'monosemantic')
        elif q > ZCRIT:
            out.append('muted' if deep else 'dead/flat')
        else:
            out.append('multimodal' if deep else 'polysemantic')
    return out

def act_np(x, name):
    if name == 'relu':
        return np.maximum(x, 0)
    if name == 'tanh':
        return np.tanh(x)
    if name == 'sigmoid':
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    return x

def analyze(Xs, model_json):
    layers = model_json['layers']
    current = Xs
    reports = []
    hidden_idx = 0
    for layer in layers:
        if layer['type'] == 'linear':
            W = np.array(layer['weight'], float)
            b = np.array(layer['bias'], float)
            z = current @ W.T + b
            if layer.get('is_output'):
                reports.append({'kind': 'output', 'preactivation': z})
                current = z
            else:
                P = influence(current, W)
                H = entropy(P)
                Z = zscore(H)
                deep = hidden_idx > 0
                reports.append({'kind': 'hidden', 'layer_index': hidden_idx + 1, 'P': P, 'H': H, 'Z': Z, 'roles': roles(Z, deep), 'preactivation': z})
                current = z
                hidden_idx += 1
        elif layer['type'] == 'activation':
            current = act_np(current, layer['name'])
            if reports and reports[-1]['kind'] == 'hidden':
                reports[-1]['activation'] = current
    return (reports, current)

def sanitize(s):
    return ''.join((c.lower() if c.isalnum() else '_' for c in str(s))).strip('_').replace('__', '_')

def real_sets():
    out = []
    specs = [('classification', 'iris', load_iris), ('classification', 'wine', load_wine), ('classification', 'breast_cancer', load_breast_cancer), ('classification', 'digits', load_digits), ('regression', 'diabetes', load_diabetes), ('regression', 'linnerud', load_linnerud)]
    for task, name, loader in specs:
        b = loader()
        X = np.asarray(b.data, float)
        y = np.asarray(b.target)
        if y.ndim > 1:
            y = y[:, 0]
        if len(X) > 500:
            rng = np.random.default_rng(SEED0)
            if task == 'classification':
                inds = []
                for c in np.unique(y):
                    ix = np.where(y == c)[0]
                    take = min(len(ix), max(20, 500 // len(np.unique(y))))
                    inds.extend(rng.choice(ix, size=take, replace=False).tolist())
                inds = np.array(sorted(inds))[:500]
            else:
                inds = rng.choice(len(X), size=500, replace=False)
            X = X[inds]
            y = y[inds]
        names = [sanitize(x) for x in getattr(b, 'feature_names', [f'x{i + 1}' for i in range(X.shape[1])])]
        out.append((task, name, X, y, names, 'canonical scikit-learn dataset'))
    return out

def synth_classification(name, seed):
    rng = np.random.default_rng(seed)
    n = 420
    if name == 'moons':
        X, y = make_moons(n_samples=n, noise=0.18, random_state=seed)
    elif name == 'circles':
        X, y = make_circles(n_samples=n, noise=0.12, factor=0.45, random_state=seed)
    elif name == 'xor':
        X = rng.normal(size=(n, 8))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    elif name == 'heavy_tail':
        X = rng.standard_t(2.4, size=(n, 12))
        y = (1.4 * X[:, 0] - 0.9 * X[:, 1] + 0.4 * X[:, 2] + rng.normal(size=n) > 0.1).astype(int)
    elif name == 'lognormal':
        X = rng.lognormal(0, 1, size=(n, 12))
        s = np.log1p(X[:, 0]) - 0.7 * np.log1p(X[:, 1]) + rng.normal(0, 0.4, n)
        y = (s > np.median(s)).astype(int)
    elif name == 'heterogeneous_scale':
        X = rng.normal(size=(n, 12)) * np.logspace(-2, 3, 12)
        s = X[:, 0] / 0.01 + X[:, 5] / np.logspace(-2, 3, 12)[5] + rng.normal(size=n)
        y = (s > 0).astype(int)
    elif name == 'zero_inflated':
        X = rng.normal(size=(n, 12))
        X[rng.random(X.shape) < 0.65] = 0
        y = (X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + rng.normal(0, 0.5, n) > 0).astype(int)
    elif name == 'outlier_contaminated':
        X = rng.normal(size=(n, 12))
        mask = rng.random(X.shape) < 0.04
        X[mask] += rng.choice([-1, 1], mask.sum()) * rng.lognormal(4, 0.7, mask.sum())
        y = (X[:, 0] + 0.7 * X[:, 1] + rng.normal(size=n) > 0).astype(int)
    elif name == 'missing10':
        X, y = make_classification(n_samples=n, n_features=12, n_informative=5, n_redundant=2, class_sep=1.0, random_state=seed)
        X[rng.random(X.shape) < 0.1] = np.nan
    else:
        params = {'balanced_easy': dict(weights=None, class_sep=1.8, flip_y=0.01, n_informative=5, n_redundant=1), 'balanced_overlap': dict(weights=None, class_sep=0.55, flip_y=0.05, n_informative=5, n_redundant=2), 'imbalanced_90_10': dict(weights=[0.9, 0.1], class_sep=1.1, flip_y=0.01, n_informative=5, n_redundant=2), 'imbalanced_98_2': dict(weights=[0.98, 0.02], class_sep=1.2, flip_y=0.005, n_informative=5, n_redundant=2), 'redundant_high': dict(weights=None, class_sep=1.0, flip_y=0.02, n_informative=3, n_redundant=7), 'informative_sparse': dict(weights=None, class_sep=1.1, flip_y=0.02, n_informative=2, n_redundant=0), 'informative_dense': dict(weights=None, class_sep=1.1, flip_y=0.02, n_informative=10, n_redundant=1), 'label_noise_10': dict(weights=None, class_sep=1.0, flip_y=0.1, n_informative=5, n_redundant=2), 'label_noise_20': dict(weights=None, class_sep=1.0, flip_y=0.2, n_informative=5, n_redundant=2), 'clusters_1': dict(weights=None, class_sep=1.0, flip_y=0.02, n_informative=5, n_redundant=2, n_clusters_per_class=1), 'clusters_3': dict(weights=None, class_sep=1.0, flip_y=0.02, n_informative=5, n_redundant=2, n_clusters_per_class=3)}
        p = params[name]
        nc = p.pop('n_clusters_per_class', 2)
        X, y = make_classification(n_samples=n, n_features=12, n_clusters_per_class=nc, random_state=seed, **p)
    return (X, np.asarray(y), [f'x{i + 1}' for i in range(X.shape[1])])

def synth_regression(name, seed):
    rng = np.random.default_rng(seed)
    n = 420
    if name == 'heavy_tail':
        X = rng.standard_t(2.4, size=(n, 12))
        y = 2 * X[:, 0] - 0.8 * X[:, 1] + np.tanh(X[:, 2]) + rng.normal(0, 0.5, n)
    elif name == 'lognormal':
        X = rng.lognormal(0, 1, size=(n, 12))
        y = 2 * np.log1p(X[:, 0]) - 0.7 * np.sqrt(X[:, 1]) + rng.normal(0, 0.3, n)
    elif name == 'heterogeneous_scale':
        scales = np.logspace(-2, 3, 12)
        X = rng.normal(size=(n, 12)) * scales
        y = 2 * X[:, 0] / scales[0] - 0.8 * X[:, 8] / scales[8] + rng.normal(0, 0.5, n)
    elif name == 'zero_inflated':
        X = rng.normal(size=(n, 12))
        X[rng.random(X.shape) < 0.65] = 0
        y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.5, n)
    elif name == 'outlier_contaminated':
        X = rng.normal(size=(n, 12))
        latent = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.4, n)
        mask = rng.random(X.shape) < 0.04
        X[mask] += rng.choice([-1, 1], mask.sum()) * rng.lognormal(4, 0.7, mask.sum())
        y = latent
    elif name == 'missing10':
        X, y = make_regression(n_samples=n, n_features=12, n_informative=6, noise=10, random_state=seed)
        X[rng.random(X.shape) < 0.1] = np.nan
    elif name == 'nonlinear_sine':
        X = rng.uniform(-3, 3, size=(n, 12))
        y = np.sin(X[:, 0]) * 2 + X[:, 1] ** 2 * 0.3 + np.cos(X[:, 2]) + rng.normal(0, 0.2, n)
    elif name == 'interaction':
        X = rng.normal(size=(n, 12))
        y = 2 * X[:, 0] * X[:, 1] + 0.7 * X[:, 2] + rng.normal(0, 0.4, n)
    elif name == 'heteroscedastic_noise':
        X = rng.normal(size=(n, 12))
        y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.2 + np.abs(X[:, 0]), n)
    else:
        params = {'linear_easy': dict(n_informative=6, noise=2), 'linear_noisy': dict(n_informative=6, noise=30), 'sparse_2': dict(n_informative=2, noise=10), 'sparse_4': dict(n_informative=4, noise=10), 'dense_10': dict(n_informative=10, noise=10), 'low_noise': dict(n_informative=6, noise=0.5), 'high_noise': dict(n_informative=6, noise=60), 'many_features': dict(n_informative=8, noise=15, n_features=24), 'few_features': dict(n_informative=3, noise=10, n_features=5)}
        p = params[name]
        nf = p.pop('n_features', 12)
        X, y = make_regression(n_samples=n, n_features=nf, random_state=seed, **p)
    return (X, np.asarray(y), [f'x{i + 1}' for i in range(X.shape[1])])

class MLP(nn.Module):

    def __init__(self, d, h1, h2, out, task):
        super().__init__()
        self.fc1 = nn.Linear(d, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out)
        self.task = task

    def forward(self, x):
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

def train_model(X, y, task, h1, h2, seed):
    strat = y if task == 'classification' else None
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=strat)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    imp = SimpleImputer(strategy='median').fit(Xtr)
    Xtr = imp.transform(Xtr)
    Xte = imp.transform(Xte)
    mean = Xtr.mean(0)
    std = Xtr.std(0)
    std[std < 1e-08] = 1
    A = (Xtr - mean) / std
    B = (Xte - mean) / std
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if task == 'classification':
        classes = np.unique(y)
        mapping = {c: i for i, c in enumerate(classes)}
        yt = np.array([mapping[v] for v in ytr])
        yev = np.array([mapping[v] for v in yte])
        out = len(classes)
        model = MLP(A.shape[1], h1, h2, out, task)
        lossfn = nn.CrossEntropyLoss()
        ty = torch.tensor(yt, dtype=torch.long)
    else:
        yt = np.asarray(ytr, float).reshape(-1, 1)
        yev = np.asarray(yte, float).reshape(-1, 1)
        out = 1
        ymean = float(np.mean(yt))
        ystd = float(np.std(yt) or 1)
        yt2 = (yt - ymean) / ystd
        model = MLP(A.shape[1], h1, h2, out, task)
        lossfn = nn.MSELoss()
        ty = torch.tensor(yt2, dtype=torch.float32)
    tx = torch.tensor(A, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=0.0001)
    model.train()
    for ep in range(55):
        opt.zero_grad()
        pred = model(tx)
        loss = lossfn(pred, ty)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(B, dtype=torch.float32)).numpy()
    if task == 'classification':
        yh = pred.argmax(1)
        metric = float(balanced_accuracy_score(yev, yh))
        metric2 = float(accuracy_score(yev, yh))
        metric_names = ('balanced_accuracy', 'accuracy')
        target_meta = {'classes': [str(c) for c in classes]}
    else:
        yp = pred[:, 0] * ystd + ymean
        yy = yev[:, 0]
        metric = float(r2_score(yy, yp))
        metric2 = float(mean_absolute_error(yy, yp))
        metric_names = ('r2', 'mae')
        target_meta = {'target_mean': ymean, 'target_std': ystd}
    Xfull = imp.transform(X)
    Xs = (Xfull - mean) / std
    layers = []
    for linear, act in [(model.fc1, 'relu'), (model.fc2, 'relu')]:
        layers.append({'type': 'linear', 'weight': linear.weight.detach().numpy().tolist(), 'bias': linear.bias.detach().numpy().tolist()})
        layers.append({'type': 'activation', 'name': act})
    layers.append({'type': 'linear', 'weight': model.out.weight.detach().numpy().tolist(), 'bias': model.out.bias.detach().numpy().tolist(), 'is_output': True})
    mj = {'schema_version': 1, 'method': 'Linear Lens', 'task': task, 'preprocessing': {'imputer_median': imp.statistics_.tolist(), 'mean': mean.tolist(), 'std': std.tolist()}, 'layers': layers, 'target': target_meta, 'training': {'seed': seed, 'epochs': 55, 'optimizer': 'Adam', 'architecture': [A.shape[1], h1, h2, out]}}
    rep, _ = analyze(Xs, mj)
    return (mj, rep, {metric_names[0]: metric, metric_names[1]: metric2})

def write_dataset(task, name, X, y, features, source):
    df = pd.DataFrame(X, columns=features)
    df['target'] = y
    path = DATA / f'{task}_{name}.csv'
    df.to_csv(path, index=False, float_format='%.10g')
    return path

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    entries = []
    for task, name, X, y, features, source in real_sets():
        entries.append((task, name, X, y, features, source))
    c_names = ['balanced_easy', 'balanced_overlap', 'imbalanced_90_10', 'imbalanced_98_2', 'redundant_high', 'informative_sparse', 'informative_dense', 'label_noise_10', 'label_noise_20', 'clusters_1', 'clusters_3', 'moons', 'circles', 'xor', 'heavy_tail', 'lognormal', 'heterogeneous_scale', 'zero_inflated', 'outlier_contaminated', 'missing10']
    r_names = ['linear_easy', 'linear_noisy', 'sparse_2', 'sparse_4', 'dense_10', 'low_noise', 'high_noise', 'many_features', 'few_features', 'nonlinear_sine', 'interaction', 'heteroscedastic_noise', 'heavy_tail', 'lognormal', 'heterogeneous_scale', 'zero_inflated', 'outlier_contaminated', 'missing10', 'quadratic_mix']
    for idx, name in enumerate(c_names):
        X, y, f = synth_classification(name, SEED0 + idx)
        entries.append(('classification', name, X, y, f, 'deterministic controlled fixture'))
    for idx, name in enumerate(r_names):
        if name == 'quadratic_mix':
            rng = np.random.default_rng(SEED0 + 100 + idx)
            X = rng.normal(size=(420, 12))
            y = 0.7 * X[:, 0] ** 2 - 1.2 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + rng.normal(0, 0.3, 420)
            f = [f'x{i + 1}' for i in range(12)]
        else:
            X, y, f = synth_regression(name, SEED0 + 100 + idx)
        entries.append(('regression', name, X, y, f, 'deterministic controlled fixture'))
    assert len(entries) == 45, len(entries)
    manifest = []
    run_rows = []
    role_rows = []
    archs = [('compact', 24, 12), ('wide', 48, 24)]
    seeds = [17, 43]
    t0 = time.time()
    for di, (task, name, X, y, features, source) in enumerate(entries, 1):
        dpath = write_dataset(task, name, X, y, features, source)
        live_model = None
        live_report = None
        for an, h1, h2 in archs:
            for seed in seeds:
                mj, rep, metrics = train_model(X, y, task, h1, h2, seed)
                if an == 'compact' and seed == 17:
                    live_model = mj
                    live_report = rep
                rec = {'dataset': name, 'task': task, 'source': source, 'architecture': an, 'seed': seed, 'rows': len(X), 'features': X.shape[1], **metrics}
                for rr in rep:
                    if rr['kind'] != 'hidden':
                        continue
                    counts = {r: rr['roles'].count(r) for r in sorted(set(rr['roles']))}
                    rec[f"layer{rr['layer_index']}_mean_entropy"] = float(np.mean(rr['H']))
                    rec[f"layer{rr['layer_index']}_effective_features"] = float(np.mean(np.exp(rr['H'])))
                    for role, count in counts.items():
                        rec[f"layer{rr['layer_index']}_{role.replace('/', '_')}_count"] = count
                    for ni, (h, z, role, p) in enumerate(zip(rr['H'], rr['Z'], rr['roles'], rr['P'])):
                        order = np.argsort(-p)[:5]
                        role_rows.append({'dataset': name, 'task': task, 'architecture': an, 'seed': seed, 'layer': rr['layer_index'], 'neuron': ni, 'role': role, 'entropy': float(h), 'zscore': float(z), 'dominant_feature': features[order[0]] if rr['layer_index'] == 1 else f"l{rr['layer_index'] - 1}_n{order[0]}", 'dominant_influence': float(p[order[0]])})
                run_rows.append(rec)
        mpath = MODELS / f'{task}_{name}.json'
        mpath.write_text(json.dumps({**live_model, 'dataset': name, 'feature_names': features}, separators=(',', ':')))
        l1 = live_report[0]
        l2 = live_report[1]
        manifest.append({'id': f'{task}_{name}', 'dataset': name, 'task': task, 'source': source, 'rows': int(len(X)), 'features': int(X.shape[1]), 'dataset_file': f'benchmarks/public_suite/datasets/{dpath.name}', 'dataset_sha256': sha(dpath), 'model_file': f'benchmarks/public_suite/models/{mpath.name}', 'model_sha256': sha(mpath), 'feature_names': features, 'live_model': 'compact · seed 17', 'paper_dataset': False, 'layer1_role_counts': {r: l1['roles'].count(r) for r in sorted(set(l1['roles']))}, 'layer2_role_counts': {r: l2['roles'].count(r) for r in sorted(set(l2['roles']))}})
        print(f'[{di:02d}/45] {task} {name}', flush=True)
    pd.DataFrame(run_rows).to_csv(RESULTS / 'benchmark_runs.csv', index=False)
    pd.DataFrame(role_rows).to_csv(RESULTS / 'neuron_roles.csv', index=False)
    rdf = pd.DataFrame(role_rows)
    st = []
    for keys, g in rdf.groupby(['dataset', 'task', 'architecture', 'layer', 'neuron']):
        if len(g) >= 2:
            st.append((*keys, int(g['role'].nunique() == 1)))
    sdf = pd.DataFrame(st, columns=['dataset', 'task', 'architecture', 'layer', 'neuron', 'stable'])
    stability = sdf.groupby(['dataset', 'task', 'architecture', 'layer'])['stable'].mean().reset_index(name='role_agreement').copy()
    stability.to_csv(RESULTS / 'role_stability.csv', index=False)
    meta = {'schema_version': 1, 'method': 'Linear Lens', 'generated_on': '2026-08-13', 'public_dataset_count': len(manifest), 'real_public_dataset_count': sum((m['source'].startswith('canonical') for m in manifest)), 'controlled_fixture_count': sum((m['source'].startswith('deterministic') for m in manifest)), 'model_analysis_runs': len(run_rows), 'neuron_records': len(role_rows), 'z_critical_one_tailed_90': ZCRIT, 'datasets': manifest, 'notes': ['The original paper evaluated ten confidential energy datasets; they are not redistributed here.', 'This public suite is additional software verification and generalization evidence, not a replacement for the paper evaluation.']}
    (ROOT / 'benchmarks/public_suite/manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    files = []
    for p in sorted((ROOT / 'benchmarks/public_suite').rglob('*')):
        if p.is_file() and p.name != 'evidence_manifest.json':
            files.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size})
    (RESULTS / 'evidence_manifest.json').write_text(json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n')
    print('DONE', len(manifest), 'datasets', len(run_rows), 'model-runs', len(role_rows), 'neurons', 'seconds', time.time() - t0)
if __name__ == '__main__':
    main()
