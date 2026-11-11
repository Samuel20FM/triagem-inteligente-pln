"""O agente de triagem (tema 1): decide destino e prioridade a partir das previsões.

É um agente BASEADO EM UTILIDADE: em vez de "classe mais provável e pronto", escolhe a acção
que maximiza a utilidade esperada, definida na tabela UTILIDADE.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

ENCAMINHAMENTO = {
    "financeiro": "Serviços Financeiros / Tesouraria",
    "matricula": "Secretaria Académica — Matrículas",
    "notas": "Secretaria Académica — Pautas e Notas",
    "infraestrutura": "Serviços de Manutenção e Infraestrutura",
}
TRIAGEM_HUMANA = "Triagem humana"

# Medida de desempenho (o "P" do PEAS), por mensagem.
UTILIDADE = {
    "certo": 1.0,             # encaminhada ao serviço correcto
    "errado": -2.0,           # encaminhada ao serviço errado (tem de ser reencaminhada)
    "humano": -0.5,           # enviada a triagem humana (custo de tempo de um funcionário)
    "urgente_perdido": -3.0,  # era urgente e não foi marcada com prioridade ALTA
}


@dataclass
class Decisao:
    accao: str        # "encaminhar" ou "triagem_humana"
    destino: str
    prioridade: str   # "ALTA" ou "NORMAL"
    motivo: str


def decidir(categoria: str, confianca: float, p_urgente: float,
            limiar_conf: float, limiar_urg: float) -> Decisao:
    prioridade = "ALTA" if p_urgente >= limiar_urg else "NORMAL"
    if confianca < limiar_conf:
        return Decisao("triagem_humana", TRIAGEM_HUMANA, prioridade,
                       f"confiança {confianca:.0%} < {limiar_conf:.0%}")
    return Decisao("encaminhar", ENCAMINHAMENTO.get(categoria, categoria), prioridade,
                   f"confiança {confianca:.0%} ≥ {limiar_conf:.0%}; P(urgente) = {p_urgente:.0%}")


def ordenar_fila(fila: pd.DataFrame) -> pd.DataFrame:
    """Prioridade ALTA primeiro; dentro de cada nível, maior P(urgente) primeiro."""
    f = fila.copy()
    f["_ordem"] = (f["prioridade"] == "ALTA").astype(int)
    return f.sort_values(["_ordem", "p_urgente"], ascending=False).drop(columns="_ordem").reset_index(drop=True)


def desempenho_agente(y_cat, pred_cat, conf, y_urg, p_urg, limiar_conf, limiar_urg) -> dict:
    """Aplica o agente a cada mensagem de teste e mede utilidade média e taxas."""
    y_cat, pred_cat = np.asarray(y_cat), np.asarray(pred_cat)
    conf, y_urg, p_urg = np.asarray(conf), np.asarray(y_urg), np.asarray(p_urg)
    humano = conf < limiar_conf
    certo = (pred_cat == y_cat) & ~humano
    errado = (pred_cat != y_cat) & ~humano
    perdido = (y_urg == 1) & (p_urg < limiar_urg)
    u = (certo * UTILIDADE["certo"] + errado * UTILIDADE["errado"]
         + humano * UTILIDADE["humano"] + perdido * UTILIDADE["urgente_perdido"])
    n_auto = int((~humano).sum())
    return {
        "utilidade": float(u.mean()),
        "taxa_humana": float(humano.mean()),
        "exatidao_encaminhadas": float(certo.sum() / n_auto) if n_auto else 0.0,
        "urgentes_perdidos": int(perdido.sum()),
        "urgentes": int((y_urg == 1).sum()),
    }


def curva_utilidade(y_cat, pred_cat, conf, y_urg, p_urg, limiar_urg) -> pd.DataFrame:
    """Utilidade média para cada limiar de confiança — o agente escolhe o melhor."""
    linhas = []
    for lim in np.round(np.arange(0.30, 0.91, 0.05), 2):
        d = desempenho_agente(y_cat, pred_cat, conf, y_urg, p_urg, lim, limiar_urg)
        linhas.append({"limiar": lim, "utilidade": d["utilidade"], "triagem humana": d["taxa_humana"]})
    return pd.DataFrame(linhas)
