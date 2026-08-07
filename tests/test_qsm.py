import numpy as np

from linearlens.qsm import influence_symbol, qsm_to_markdown, qualitative_symbolic_matrix


def test_symbol_bands() -> None:
    assert influence_symbol(0.61) == "★★★"
    assert influence_symbol(0.31) == "★★"
    assert influence_symbol(0.11) == "★"
    assert influence_symbol(0.09) == "◦"


def test_qsm_markdown() -> None:
    matrix = qualitative_symbolic_matrix(np.array([[0.7, 0.2, 0.1]]))
    text = qsm_to_markdown(matrix, ["a", "b", "c"], np.array(["mono"], dtype=object))
    assert "★★★" in text
    assert "mono" in text
