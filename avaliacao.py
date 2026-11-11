"""Leitura do corpus, divisão treino/teste sem fuga, treino das 6 combinações e métricas (temas 9 e 13)."""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

from classificadores import CLASSIFICADORES, criar_pipeline
from representacoes import REPRESENTACOES
from texto import normalizar

COLUNAS_OBRIGATORIAS = {"mensagem", "categoria"}
MIN_TEXTOS_POR_CATEGORIA = 5   # precisa de chegar para os 5 folds da validação cruzada
FOLDS_CV, REPETICOES_CV = 5, 3


# ------------------------------------------------------------------ DADOS
def ler_csv(origem) -> pd.DataFrame:
    """Lê o CSV (separador ',' ou ';' detectado), limpa e valida colunas.

    Lança ValueError com uma mensagem legível se algo falhar.
    """
    try:
        df = pd.read_csv(origem, sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Não foi possível ler o ficheiro como CSV ({e}).") from e
    df.columns = [str(c).strip().lower() for c in df.columns]
    falta = COLUNAS_OBRIGATORIAS - set(df.columns)
    if falta:
        raise ValueError(f"O CSV não tem a(s) coluna(s): {', '.join(sorted(falta))}.")
    # remover vazios ANTES de converter para texto (senão NaN vira a palavra "nan")
    df = df.dropna(subset=["mensagem", "categoria"]).copy()
    df["mensagem"] = df["mensagem"].astype(str).str.strip()
    df["categoria"] = df["categoria"].astype(str).str.strip().str.lower()
    return df[df["mensagem"] != ""].reset_index(drop=True)


def validar_corpus(df: pd.DataFrame) -> list:
    """Lista de problemas que impedem um treino/avaliação com sentido (vazia = tudo bem)."""
    problemas = []
    if df["categoria"].nunique() < 2:
        problemas.append("O corpus precisa de pelo menos 2 categorias.")
    distintos = df.groupby("categoria")["mensagem"].apply(lambda s: s.map(normalizar).nunique())
    poucas = distintos[distintos < MIN_TEXTOS_POR_CATEGORIA]
    if len(poucas):
        problemas.append(
            f"Cada categoria precisa de ≥ {MIN_TEXTOS_POR_CATEGORIA} textos distintos. Em falta: "
            + ", ".join(f"{c} ({n})" for c, n in poucas.items()) + "."
        )
    return problemas


# ------------------------------------------------------------------ DIVISÃO
def dividir(y, grupos, proporcao_teste: float, seed: int):
    """Hold-out AGRUPADO (o mesmo texto nunca fica nos dois lados) e ESTRATIFICADO (mesmas
    proporções de categorias no treino e no teste). Usa o 1.º fold de um StratifiedGroupKFold
    com k = 1 / proporção, por isso o teste fica com ≈ a proporção pedida de LINHAS.
    """
    k = max(2, round(1 / proporcao_teste))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return next(cv.split(np.zeros(len(y)), y, grupos))


# ------------------------------------------------------------------ MÉTRICAS
def _metricas(y_true, y_pred, classes):
    rep = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": rep["macro avg"]["precision"],
        "rec": rep["macro avg"]["recall"],
        "f1": rep["macro avg"]["f1-score"],
        "por_classe": {c: rep[c] for c in classes},
        "cm": confusion_matrix(y_true, y_pred, labels=classes),
    }


def calibracao(y_true, proba, classes, n_bins: int = 10) -> dict:
    """Mede se a confiança do modelo corresponde à taxa real de acerto.

    ECE (erro de calibração esperado) = Σ (n_b / N) · |exatidão_b − confiança_b|.
    Brier = média de Σ_k (p_k − 1[classe = k])².
    """
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    conf = proba.max(axis=1)
    acerto = (classes[proba.argmax(axis=1)] == y_true).astype(float)
    bordas = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bordas[1:-1]), 0, n_bins - 1)
    linhas, ece = [], 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            c_b, a_b = conf[sel].mean(), acerto[sel].mean()
            ece += sel.mean() * abs(a_b - c_b)
            linhas.append({"confiança média": c_b, "exatidão": a_b, "n": int(sel.sum())})
    one_hot = (y_true[:, None] == classes[None, :]).astype(float)
    brier = float(((proba - one_hot) ** 2).sum(axis=1).mean())
    return {"ece": float(ece), "brier": brier, "tabela": pd.DataFrame(linhas)}


# ------------------------------------------------------------------ TREINO
# 1 thread de BLAS: com várias, as cópias do OpenBLAS (numpy + scipy) competem e o treino fica ~10× mais lento.
@threadpool_limits.wrap(limits=1)
def treinar(df: pd.DataFrame, usar_stop=True, ngram=(1, 1), proporcao_teste=0.2, seed=42,
            alpha=1.0, C=5.0) -> dict:
    """Treina e avalia as 3 representações × 2 classificadores.

    Para cada combinação: hold-out sem fuga, hold-out COM fuga (só para comparação),
    validação cruzada agrupada 5 folds × 3 sementes e calibração.
    """
    X, y = df["mensagem"], df["categoria"]
    grupos = X.map(normalizar)          # mensagens iguais ficam sempre do mesmo lado
    classes = sorted(y.unique())
    tr, te = dividir(y, grupos, proporcao_teste, seed)
    tr_n, te_n = next(StratifiedShuffleSplit(1, test_size=len(te) / len(df), random_state=seed).split(X, y))

    out = {"classes": classes, "idx_treino": tr, "idx_teste": te,
           "teste": df.iloc[te].reset_index(drop=True),
           "n_treino": len(tr), "n_teste": len(te), "modelos": {}}

    for rep in REPRESENTACOES:
        for clf in CLASSIFICADORES:
            base = criar_pipeline(rep, clf, usar_stop, ngram, alpha, C)

            pipe = clone(base)
            t0 = time.perf_counter()
            pipe.fit(X.iloc[tr], y.iloc[tr])
            t_treino = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            proba = pipe.predict_proba(X.iloc[te])
            t_inf = (time.perf_counter() - t0) * 1000 / len(te)
            pred = pipe.classes_[proba.argmax(axis=1)]

            m = _metricas(y.iloc[te], pred, classes)
            m.update(pipe=pipe, pred=pred, proba=proba, t_treino=t_treino, t_inf=t_inf,
                     vocab=len(pipe.named_steps["vec"].vocabulary_),
                     calib=calibracao(y.iloc[te], proba, pipe.classes_))

            # com fuga de dados (split aleatório ingénuo) — só para mostrar o efeito
            p_n = clone(base).fit(X.iloc[tr_n], y.iloc[tr_n])
            m["fuga_acc"] = accuracy_score(y.iloc[te_n], p_n.predict(X.iloc[te_n]))

            # validação cruzada agrupada e estratificada, repetida com 3 sementes
            accs, f1s = [], []
            for s in range(REPETICOES_CV):
                skf = StratifiedGroupKFold(FOLDS_CV, shuffle=True, random_state=seed + s)
                for a, b in skf.split(X, y, grupos):
                    pr = clone(base).fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b])
                    accs.append(accuracy_score(y.iloc[b], pr))
                    f1s.append(f1_score(y.iloc[b], pr, average="macro"))
            m.update(cv_acc=(np.mean(accs), np.std(accs)), cv_f1=(np.mean(f1s), np.std(f1s)))
            out["modelos"][(rep, clf)] = m

    out["melhor"] = max(out["modelos"], key=lambda k: out["modelos"][k]["cv_f1"][0])
    return out


def tabela_resultados(R: dict) -> pd.DataFrame:
    """Tabela no formato do Guia: ficheiro · método · conjunto · métricas."""
    linhas = []
    for (rep, clf), m in R["modelos"].items():
        linhas.append({
            "Ficheiro de referência": "corpus_referencia",
            "Representação": REPRESENTACOES[rep],
            "Classificador": CLASSIFICADORES[clf],
            "Vocabulário": m["vocab"],
            "Exatidão (hold-out)": round(m["acc"] * 100, 1),
            "Precisão macro": round(m["prec"] * 100, 1),
            "Revocação macro": round(m["rec"] * 100, 1),
            "F1 macro (hold-out)": round(m["f1"] * 100, 1),
            "F1 macro CV (média)": round(m["cv_f1"][0] * 100, 1),
            "F1 macro CV (dp)": round(m["cv_f1"][1] * 100, 1),
            "ECE": round(m["calib"]["ece"], 3),
            "Treino (ms)": round(m["t_treino"], 1),
        })
    return pd.DataFrame(linhas)
