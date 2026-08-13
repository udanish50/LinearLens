# Linear Lens public software evidence

The peer-reviewed Linear Lens study evaluated ten real-world energy datasets, but the publication reports those data as confidential. This repository therefore does **not** redistribute or imitate them as if they were the paper data.

The `benchmarks/public_suite/` directory is an additional software-verification suite designed for transparent, reproducible inspection of the released implementation and browser demonstrator. It contains:

- **45 datasets**: 6 canonical public scikit-learn datasets and 39 deterministic controlled fixtures;
- **180 trained MLP analyses** across two widths and two independent training seeds;
- **9,720 neuron records** with entropy, z-score, role, dominant source, and influence magnitude;
- **36 controlled semantic-recovery scenarios** spanning Gaussian, lognormal, Student-t, zero-inflated, correlated, and heterogeneous-scale inputs;
- **45 fixed-model bootstrap tests** (40 resamples each) to measure sensitivity of role assignment to the analyzed sample;
- SHA-256 hashes for every browser-verification dataset and exported model.

## What is being tested

The suite tests numerical and behavioral properties of Linear Lens: normalized influence distributions, finite entropy/z-scores, role assignment, reproducible browser export, role recovery on controlled structures, and sample-resampling agreement. Prediction metrics are included to document the analyzed models; they are **not** presented as an XAI quality score.

## What is not being claimed

This suite does not prove universal superiority over other explanation methods. Linear Lens answers a mechanistic neuron/layer question that is not identical to local feature-attribution methods such as Integrated Gradients or perturbation importance. The website therefore compares their purpose, intervention requirements, and outputs rather than fabricating a single cross-method score.

## Rebuild

Run `generate_public_suite.py` to recreate the datasets, trained dense models, and base analysis ledger. Then run `build_validation_suites.py` to rebuild the controlled semantic-recovery and fixed-model bootstrap checks. `verify_public_suite.py` is the lightweight CI verifier and does not retrain models.
