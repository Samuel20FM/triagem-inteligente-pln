import io

import numpy as np
import pandas as pd
import pytest

from avaliacao import calibracao, dividir, ler_csv, tabela_resultados, validar_corpus
from texto import normalizar


def test_corpus_tem_colunas_tamanho_e_categorias(df):
    assert {"mensagem", "categoria", "urgente", "registo"} <= set(df.columns)
    assert 300 <= len(df) <= 800
    assert set(df["categoria"]) == {"financeiro", "matricula", "notas", "infraestrutura"}
    assert validar_corpus(df) == []


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_divisao_sem_fuga_estratificada_e_com_proporcao_certa(df, seed):
    g = df["mensagem"].map(normalizar)
    tr, te = dividir(df["categoria"], g, 0.2, seed)
    assert set(g.iloc[tr]).isdisjoint(set(g.iloc[te]))             # sem fuga
    assert 0.17 <= len(te) / len(df) <= 0.23                       # ≈ 20 % das linhas
    p_tot = df["categoria"].value_counts(normalize=True)
    p_te = df["categoria"].iloc[te].value_counts(normalize=True)
    assert (p_te - p_tot).abs().max() < 0.05                       # estratificado


def test_resultados_coerentes(R, df):
    assert len(R["modelos"]) == 6
    for m in R["modelos"].values():
        assert 0.5 < m["acc"] <= 1 and 0.5 < m["cv_f1"][0] <= 1
        assert m["cm"].sum() == R["n_teste"]
        assert 0 <= m["calib"]["ece"] <= 1
    assert R["n_treino"] + R["n_teste"] == len(df)
    assert R["melhor"] in R["modelos"]


def test_tabela_resultados_formato_do_guia(R):
    t = tabela_resultados(R)
    assert len(t) == 6 and t.columns[0] == "Ficheiro de referência"


def test_calibracao_perfeita_tem_ece_zero():
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = calibracao(["a", "b"], proba, ["a", "b"])
    assert c["ece"] == pytest.approx(0) and c["brier"] == pytest.approx(0)


@pytest.mark.parametrize("texto", ["mensagem;categoria\nola;notas\n", "mensagem,categoria\nola,notas\n"])
def test_ler_csv_aceita_virgula_e_ponto_e_virgula(texto):
    assert list(ler_csv(io.StringIO(texto))["mensagem"]) == ["ola"]


def test_ler_csv_remove_vazios_sem_criar_nan():
    d = ler_csv(io.StringIO("mensagem,categoria\n,financeiro\nola,notas\n"))
    assert list(d["mensagem"]) == ["ola"]


def test_ler_csv_sem_coluna_da_erro_claro():
    with pytest.raises(ValueError, match="categoria"):
        ler_csv(io.StringIO("mensagem,classe\nola,a\n"))


def test_validar_recusa_uma_categoria_e_poucos_textos(df):
    assert validar_corpus(df[df["categoria"] == "notas"])
    assert validar_corpus(pd.DataFrame({"mensagem": ["a", "b", "c"], "categoria": ["x", "y", "x"]}))
