# Model / Method Card — Linear Lens

## Intended use

Linear Lens is an observational interpretability procedure for trained neural-network components where the relevant observed inputs/activations and forward weights are accessible.

## Intended users

Researchers, model auditors, ML engineers, and domain experts studying how learned representations relate to input features and upstream activations.

## Non-interventional scope

The analysis computes statistics from the trained model's existing values. It does not perturb inputs, patch activations, or update parameters.

## Limitations

- A high influence score is an interpretability statistic, not a general proof of causality.
- The entropy role labels are layer-relative.
- Architecture adapters expose selected components; complex architectures may require custom adapters.
- Pre-activation analysis is most directly interpretable for components with explicit linear weight matrices.
- Exact reproduction of the paper's private-data experiments is impossible without authorized access to those data.
- Equation 13 and the paper's controlled synthetic result table contain a numerical labeling inconsistency; this repository follows Equation 13 by default and documents the mismatch.

## High-stakes use

Do not use a Linear Lens explanation alone as the basis for medical, legal, employment, credit, safety, or other high-stakes decisions. Treat the output as one model-auditing signal among broader validation and governance procedures.
