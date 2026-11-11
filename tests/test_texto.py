import pytest

from texto import STOPWORDS_PT, normalizar


@pytest.mark.parametrize("entrada,esperado", [
    ("Matrícula!", "matricula"),
    ("  PAGAMENTO   da Propina?? ", "pagamento da propina"),
    ("Não consigo", "nao consigo"),
    ("", ""),
])
def test_normalizar(entrada, esperado):
    assert normalizar(entrada) == esperado


def test_negacao_nao_e_stopword():
    assert {"nao", "sem", "nem"}.isdisjoint(STOPWORDS_PT)
