# Public verification suite

This suite is additional software evidence for the open-source Linear Lens implementation. It is intentionally separate from the peer-reviewed paper's original ten energy datasets, which the publication reports as confidential.

## Evidence layers

1. **45 public/controlled datasets** with one browser model per dataset. The website verifies the dataset and model hashes before recomputing dense-network Linear Lens analysis.
2. **180 trained-model analyses**: two dense architectures × two training seeds across the 45 datasets.
3. **36 controlled role-recovery scenarios** with known concentrated, distributed, and flat neuron structures. These test whether the entropy/z-score decision pipeline recovers known roles under several input distributions and dimensions.
4. **45 fixed-model bootstrap studies** with 40 resamples each. These keep network weights fixed and measure sensitivity of the global role summary to the analyzed observations.
5. **Neuron-level ledger** with entropy, z-score, role, dominant source, and influence magnitude.

## Important interpretation note about cross-seed neuron indices

`results/role_stability.csv` is a diagnostic produced by the base generator that compares the *same numeric neuron index* across independently trained models. It must **not** be interpreted as a formal stability score: hidden neurons are permutation-symmetric and independently trained networks can reorder or reorganize internal representations. The website therefore does not headline this diagnostic. The fixed-model bootstrap analysis is the appropriate sample-resampling stability check in this public suite.

## Rebuild

```bash
python benchmarks/public_suite/generate_public_suite.py
python benchmarks/public_suite/build_validation_suites.py
python benchmarks/public_suite/verify_public_suite.py
```

The generator is deterministic at the declared seeds but retraining can still show small platform-dependent floating-point variation. Tracked hashes correspond to the committed artifacts.
