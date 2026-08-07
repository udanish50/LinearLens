# Reproducibility notes

## What is directly specified by the paper

The repository treats the following as publication-grounded details:

- input-to-neuron influence based on the average absolute feature × weight product;
- row-normalized influence distributions;
- Shannon entropy and layer-relative entropy z-scores;
- Equation 13 cutoffs at `-1.645` and `+1.645` for first-layer role assignment;
- regression validation for polysemantic neurons, with `0.15` provided as an example influence threshold;
- QSM / symbolic pathway summaries;
- reuse of influence, normalization, entropy, and z-score logic for deeper layers;
- MLP, LSTM-gate, and Transformer-query analyses as representative architecture components;
- 60/20/20 temporal split, Min-Max scaling, 24-hour windows, stride 1, and 24-hour forecasting in the energy evaluation;
- 64, 128, 256, and 512 unit model widths;
- a confidential-data policy.

## Important numerical inconsistency in the publication

Equation 13 states:

- `z < -1.645` → monosemantic;
- `-1.645 <= z <= 1.645` → polysemantic;
- `z > 1.645` → dead.

However, the controlled synthetic results table reports examples such as `z=-0.8833` and `z=-0.7799` as monosemantic and `z=1.6153` as dead. Those labels do not satisfy the Equation 13 thresholds.

This repository does **not** silently rewrite either statement. The default `role_mode="zscore"` follows Equation 13 exactly. An optional `role_mode="illustrative"` uses dominance/diffusion shares to support qualitative experiments resembling the illustrative examples, but it is explicitly labeled as a convenience rule rather than Equation 13.

## Engineering choices not fully specified in the article

These are implementation choices and should not be presented as recovered original-source settings:

- numerical epsilon values;
- exact symbolic threshold implementation around the paper's approximate 60/30/10 strengths;
- handling of zero-sum influence rows;
- which PyTorch module API is used to expose LSTM gates or attention query weights;
- storage, CLI, testing, and software packaging conventions.

## Reproducing the real-world evaluation

The paper's underlying real-world data are confidential. Therefore exact result reproduction is not possible from this public repository alone. Users with authorized data can use the documented preprocessing protocol and adapters to run the method on their own trained models.
