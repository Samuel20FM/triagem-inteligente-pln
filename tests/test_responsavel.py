import numpy as np
import pandas as pd
import pytest

from responsavel import (anonimizar, limiares_por_subgrupo, metricas_por_subgrupo,
                         metricas_urgencia_por_subgrupo, pesos_reponderacao, taxa_urgencia_a_categoria_igual)


@pytest.mark.parametrize("texto,tipo", [
    ("liguem para +258 84 123 4567", "TELEFONE"),
    ("o meu numero e 841234567", "TELEFONE"),
    ("email joao.m@up.ac.mz", "EMAIL"),
    ("BI 110100123456B", "BI"),
    ("sou o estudante 20231234", "NUMERO_ID"),
])
def test_anonimizar_detecta(texto, tipo):
    mascarado, enc = anonimizar(texto)
    assert f"[{tipo}]" in mascarado and enc[0][0] == tipo


def test_anonimizar_nao_mexe_em_texto_normal():
    t = "o projector da sala 14 nao funciona desde 2026"
    assert anonimizar(t) == (t, [])


def test_subgrupos_somam_ao_total(R):
    m = R["modelos"][R["melhor"]]
    s = metricas_por_subgrupo(R["teste"], m["pred"])
    assert s["n"].sum() == R["n_teste"]
    u = metricas_urgencia_por_subgrupo(R["teste"], np.full(R["n_teste"], 0.6), 0.5)
    assert (u["urgentes (previsto)"] == 1).all()


def test_rotulo_historico_mais_urgente_em_formal(df):
    t = taxa_urgencia_a_categoria_igual(df)
    assert (t["formal"] > t["informal"]).all()


def test_reponderacao_iguala_taxas():
    g = np.array(["f"] * 10 + ["i"] * 10)
    y = np.array([1] * 6 + [0] * 4 + [1] * 2 + [0] * 8)
    w = pesos_reponderacao(g, y)
    d = pd.DataFrame({"g": g, "y": y, "w": w})
    taxas = d.groupby("g").apply(lambda p: (p.w * p.y).sum() / p.w.sum(), include_groups=False)
    assert taxas["f"] == pytest.approx(taxas["i"])


def test_limiares_por_subgrupo_atingem_revocacao():
    g = np.array(["f"] * 4 + ["i"] * 4)
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    p = np.array([0.45, 0.40, 0.1, 0.1, 0.8, 0.7, 0.2, 0.2])
    lims = limiares_por_subgrupo(g, y, p, revocacao_alvo=1.0)
    assert lims["f"] == pytest.approx(0.40) and lims["i"] == pytest.approx(0.50)
