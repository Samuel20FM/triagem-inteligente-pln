import pandas as pd

from agente import TRIAGEM_HUMANA, decidir, desempenho_agente, ordenar_fila


def test_pouca_confianca_vai_para_humano():
    d = decidir("notas", 0.40, 0.9, limiar_conf=0.55, limiar_urg=0.5)
    assert d.accao == "triagem_humana" and d.destino == TRIAGEM_HUMANA and d.prioridade == "ALTA"


def test_encaminha_com_prioridade():
    d = decidir("financeiro", 0.90, 0.2, 0.55, 0.5)
    assert d.accao == "encaminhar" and "Financeiros" in d.destino and d.prioridade == "NORMAL"


def test_fila_ordenada_alta_primeiro():
    f = pd.DataFrame({"prioridade": ["NORMAL", "ALTA", "ALTA"], "p_urgente": [0.4, 0.6, 0.9]})
    assert list(ordenar_fila(f)["p_urgente"]) == [0.9, 0.6, 0.4]


def test_utilidade():
    d = desempenho_agente(["a", "b"], ["a", "a"], [0.9, 0.9], [0, 1], [0.1, 0.1], 0.5, 0.5)
    # certo (+1) ; errado (−2) + urgente perdido (−3) → média −2
    assert d["utilidade"] == -2.0 and d["urgentes_perdidos"] == 1
