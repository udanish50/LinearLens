import torch

from linearlens import analyze_linear_component
from linearlens.adapters import lstm_gate_weight, multihead_attention_query_weight


def main() -> None:
    torch.manual_seed(5)
    x = torch.randn(128, 5)

    lstm = torch.nn.LSTM(input_size=5, hidden_size=16, batch_first=True)
    gate = lstm_gate_weight(lstm, gate="input")
    gate_report = analyze_linear_component(x, gate)
    print("LSTM input-gate roles:", gate_report.roles.tolist())

    attention = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
    q_weight = multihead_attention_query_weight(attention)
    q_inputs = torch.randn(128, 8)
    q_report = analyze_linear_component(q_inputs, q_weight)
    print("Transformer query roles:", q_report.roles.tolist())


if __name__ == "__main__":
    main()
