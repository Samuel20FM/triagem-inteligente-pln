"""Segundo alvo: prever a coluna `urgente` (0/1) para priorizar a fila (temas 9 e 1)."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from threadpoolctl import threadpool_limits

from classificadores import criar_pipeline
from texto import normalizar

# Escolhido por validação cruzada agrupada (F1 macro ≈ 75 %, ver IMPLEMENTACAO.md).
REP_URGENCIA, CLF_URGENCIA = "tfidf", "nb"


@threadpool_limits.wrap(limits=1)
def treinar_urgencia(df: pd.DataFrame, idx_treino, idx_teste, usar_stop=True, ngram=(1, 1),
                     alpha=1.0, pesos=None) -> dict:
    """Treina o modelo de urgência com o MESMO split da categoria (sem fuga).

    `pesos` (opcional, um por linha de df) entra como sample_weight — usado na reponderação.
    """
    X, y = df["mensagem"], df["urgente"].astype(int)
    pipe = criar_pipeline(REP_URGENCIA, CLF_URGENCIA, usar_stop, ngram, alpha)
    params = {}
    if pesos is not None:
        params["clf__sample_weight"] = np.asarray(pesos)[idx_treino]
    pipe.fit(X.iloc[idx_treino], y.iloc[idx_treino], **params)
    p = pipe.predict_proba(X.iloc[idx_teste])[:, list(pipe.classes_).index(1)]
    return {"pipe": pipe, "p_teste": p, "y_teste": y.iloc[idx_teste].to_numpy()}


@threadpool_limits.wrap(limits=1)
def prob_fora_da_amostra(df: pd.DataFrame, idx_treino, usar_stop=True, ngram=(1, 1), alpha=1.0,
                         seed: int = 42) -> np.ndarray:
    """P(urgente) de cada mensagem de TREINO, prevista por um modelo que não a viu (CV agrupada).

    Serve para escolher limiares sem espreitar o conjunto de teste.
    """
    d = df.iloc[idx_treino]
    grupos = d["mensagem"].map(normalizar)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(criar_pipeline(REP_URGENCIA, CLF_URGENCIA, usar_stop, ngram, alpha),
                              d["mensagem"], d["urgente"].astype(int), groups=grupos, cv=cv,
                              method="predict_proba")
    return proba[:, 1]


def prob_urgente(pipe, texto: str) -> float:
    return float(pipe.predict_proba([texto])[0][list(pipe.classes_).index(1)])


def metricas_urgencia(y_true, p, limiar: float = 0.5) -> dict:
    """Métricas da classe 'urgente'. A revocação é a mais importante: um urgente perdido custa mais."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(p) >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": (tp + tn) / len(y_true), "prec": prec, "rec": rec, "f1": f1,
            "cm": np.array([[tn, fp], [fn, tp]]), "alarmes": int(tp + fp)}


def curva_limiar(y_true, p) -> pd.DataFrame:
    """Precisão e revocação dos urgentes para limiares de 0,05 a 0,95."""
    linhas = []
    for lim in np.round(np.arange(0.05, 0.96, 0.05), 2):
        m = metricas_urgencia(y_true, p, lim)
        linhas.append({"limiar": lim, "precisão": m["prec"], "revocação": m["rec"]})
    return pd.DataFrame(linhas)
