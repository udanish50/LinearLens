# Software architecture

```text
Observed inputs / activations + trained forward weights
                    |
                    v
        Mean absolute influence (Eq. 9)
                    |
                    v
          Row normalization (Eq. 10)
                    |
                    v
       Shannon entropy + layer z-score
                    |
          +---------+---------+
          |                   |
          v                   v
  Neuron-role labels     Polysemantic OLS
          |                   |
          +---------+---------+
                    |
                    v
                    QSM
                    |
                    v
        Repeat on deeper activations
                    |
                    v
       Semantic / compositional trace
```

The implementation never writes into model parameters. The optional PyTorch recorder uses ordinary forward hooks that return `None`, so outputs are not replaced.
