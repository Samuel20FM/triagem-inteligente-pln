import numpy as np

from urgencia import curva_limiar, metricas_urgencia, prob_fora_da_amostra, prob_urgente, treinar_urgencia


def test_urgencia_treina_e_preve(df, R):
    U = treinar_urgencia(df, R["idx_treino"], R["idx_teste"])
    assert len(U["p_teste"]) == R["n_teste"] and ((U["p_teste"] >= 0) & (U["p_teste"] <= 1)).all()
    m = metricas_urgencia(U["y_teste"], U["p_teste"])
    assert m["cm"].sum() == R["n_teste"] and m["rec"] > 0.4
    assert prob_urgente(U["pipe"], "Falta de energia no bloco onde decorre o exame de hoje. Solicito intervencao imediata") > 0.5


def test_limiar_mais_baixo_aumenta_revocacao(df, R):
    U = treinar_urgencia(df, R["idx_treino"], R["idx_teste"])
    c = curva_limiar(U["y_teste"], U["p_teste"])
    assert c["revocação"].is_monotonic_decreasing


def test_previsao_fora_da_amostra_so_usa_treino(df, R):
    p = prob_fora_da_amostra(df, R["idx_treino"])
    assert len(p) == R["n_treino"] and np.isfinite(p).all()
