# Linear Lens

**Reference implementation of Linear Lens: a human-centered, non-interventional mechanistic interpretability workflow for neural networks.**

[![CI](https://github.com/udanish50/LinearLens/actions/workflows/ci.yml/badge.svg)](https://github.com/udanish50/LinearLens/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

Linear Lens analyzes a trained model **without perturbing inputs, patching activations, or changing learned parameters**. It works from observed inputs/activations and the model's existing forward-computation weights to quantify feature-to-neuron influence, classify functional neuron roles, summarize layers with a Qualitative Symbolic Matrix (QSM), and trace semantic influence through deeper layers.

> **Publication**  
> Muhammad Umair Danish, Memoona Aziz, Umair Rehman, Katarina Grolinger,  
> “Linear Lens: A human-centered, non-interventional mechanistic approach to explainable AI,”  
> *Machine Learning with Applications*, 24 (2026), 100897.  
> DOI: `10.1016/j.mlwa.2026.100897`

## Why this repository exists

The paper reports the method and its evaluation, but the research datasets and participant data are confidential. This repository therefore provides a clean, reusable implementation of the published procedure plus synthetic examples. **It does not claim to be the authors' undisclosed original experimental source tree, and it does not reproduce confidential data.** Where the article leaves engineering choices unspecified, the repository marks them as implementation defaults.

## Core workflow

1. **Phase 1 — neuron discovery**
   - compute mean absolute input–weight influence per feature and neuron;
   - normalize influences into a probability-style distribution;
   - compute Shannon entropy and layer-relative z-scores;
   - classify first-layer units as monosemantic, polysemantic, or dead/diffuse.
2. **Phase 2 — layer interpretation**
   - validate polysemantic feature sets using non-interventional linear regression;
   - produce a human-readable QSM using configurable symbolic strength bands.
3. **Phase 3 — deeper-layer tracing**
   - repeat the same influence/entropy pipeline using upstream activations as inputs;
   - label deeper units as unimodal, multimodal, or muted;
   - trace compositional pathways toward outputs.
4. **Phase 4 — human evaluation support**
   - this repository includes only aggregate-analysis utilities and documentation;
   - no confidential participant responses are included.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Quick start

Run the controlled synthetic example inspired by the article's known-ground-truth experiment:

```bash
linear-lens synthetic
```

Analyze a PyTorch linear layer without changing the model:

```python
import torch
from linearlens import analyze_linear_component

x = torch.randn(256, 5)
layer = torch.nn.Linear(5, 16)
report = analyze_linear_component(
    x,
    layer.weight,
    feature_names=["temperature", "hour", "energy", "day_of_year", "day_of_week"],
)

print(report.roles)
print(report.qsm_markdown())
```

Capture inputs to selected `torch.nn.Linear` layers during an ordinary inference pass:

```python
from linearlens.capture import ActivationRecorder

with ActivationRecorder(model, module_types=(torch.nn.Linear,)) as recorder:
    _ = model(batch)

for name, snapshot in recorder.snapshots.items():
    print(name, snapshot.inputs.shape)
```

The hook returns no replacement value and therefore does not alter the forward computation.

## PyTorch architecture adapters

Linear Lens was evaluated on MLP, LSTM, and Transformer models. The package includes helpers for representative forward-computation components:

```python
from linearlens.adapters import lstm_gate_weight, multihead_attention_query_weight

w_input_gate = lstm_gate_weight(lstm, gate="input")
w_forget_gate = lstm_gate_weight(lstm, gate="forget")
w_query = multihead_attention_query_weight(attention)
```

PyTorch LSTM gate ordering is handled explicitly as input, forget, cell, output.

## Paper-faithful defaults

The default z-score role classifier uses the article's Equation 13 thresholds:

```text
z < -1.645       -> monosemantic
-1.645 <= z <= 1.645 -> polysemantic
z > 1.645        -> dead / diffuse
```

For deeper layers the same entropy pipeline is mapped to:

```text
low entropy  -> unimodal
middle       -> multimodal
high entropy -> muted
```

The paper also presents illustrative dominance examples and a controlled synthetic table whose reported labels are not numerically consistent with Equation 13. This repository **does not silently hide that discrepancy**. See [`docs/reproducibility.md`](docs/reproducibility.md) and the optional `illustrative` classifier mode.

## Data policy

No real energy-consumption data, participant-level user-study responses, private building data, or trained research checkpoints are distributed here. The article states that the data used are confidential. `scripts/validate_release.py` checks the repository for common accidental data/secret artifacts before publishing.

## Repository structure

```text
src/linearlens/        Core implementation
examples/              Synthetic and PyTorch demonstrations
tests/                 Unit and behavior tests
docs/                  Method, architecture, reproducibility, ethics, results
assets/                 Original repository diagrams
scripts/               Release validation and GitHub publishing
```

## Citation

Please cite the publication if this implementation supports your research. See [`CITATION.cff`](CITATION.cff) and [`paper.bib`](paper.bib).

## License

The code in this repository is released under the MIT License. The associated journal article is a separate publication with its own publisher license; the article PDF is intentionally not redistributed in this repository.
