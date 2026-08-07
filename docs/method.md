# Method mapping

This implementation is organized around the four phases described in the Linear Lens publication.

## Phase 1 — neuron discovery

For observed inputs `X` and a forward-computation weight matrix `W`, the package computes

`mu[i,j] = mean_k |X[k,j] * W[i,j]|`

and then normalizes each neuron's row to obtain a probability-style feature influence distribution. Shannon entropy is computed per neuron, followed by a layer-relative z-score. The default classifier implements Equation 13 with cutoffs at ±1.645.

## Phase 2 — polysemantic validation and QSM

Polysemantic neurons can be behaviorally validated by fitting an ordinary least-squares regression from their influential input subset to the observed pre-activation. The default feature inclusion threshold is 0.15 because the paper gives this value as an example; it is configurable.

The Qualitative Symbolic Matrix converts influence values to symbols. The repository defaults are derived from the paper's approximate pathway strengths: 60% = very strong, 30% = strong, 10% = moderate, below 10% = weak/negligible. These are visualization defaults, not learned parameters.

## Phase 3 — deeper layers

For a deeper linear component, the upstream activations become the input matrix. The same influence, normalization, entropy, and z-score logic is used. Low-, intermediate-, and high-entropy regions are represented as unimodal, multimodal, and muted, respectively. Output components can then be summarized as compositional pathways.

## Phase 4 — human evaluation

The paper evaluates cognitive load, trust/usability, comprehensibility, and actionability with a 400-participant study. This repository intentionally excludes participant-level responses. It contains only general statistical utilities that can be applied to appropriately authorized data.
