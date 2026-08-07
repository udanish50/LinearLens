import torch

from linearlens.adapters import lstm_gate_weight, multihead_attention_query_weight


def test_lstm_gate_shapes() -> None:
    lstm = torch.nn.LSTM(input_size=5, hidden_size=7)
    for gate in ("input", "forget", "cell", "output"):
        assert lstm_gate_weight(lstm, gate=gate).shape == (7, 5)


def test_attention_query_shape() -> None:
    attn = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2)
    assert multihead_attention_query_weight(attn).shape == (8, 8)
