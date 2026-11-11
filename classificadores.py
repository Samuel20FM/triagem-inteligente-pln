"""Classificadores (temas 8/9): Naive Bayes multinomial e Regressão Logística, com explicação por termo."""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from representacoes import criar_vectorizador

CLASSIFICADORES = {"nb": "Naive Bayes", "lr": "Regressão Logística"}


def criar_classificador(chave: str, alpha: float = 1.0, C: float = 5.0):
    if chave == "nb":
        return MultinomialNB(alpha=alpha)
    if chave == "lr":
        # C = inverso da regularização (C grande → menos regularização).
        return LogisticRegression(C=C, max_iter=3000)
    raise ValueError(f"Classificador desconhecido: {chave}")


def criar_pipeline(rep: str, clf: str, usar_stop: bool = True, ngram: tuple = (1, 1),
                   alpha: float = 1.0, C: float = 5.0) -> Pipeline:
    """Vectorizador + classificador num Pipeline: o vocabulário/IDF só se aprende no treino."""
    return Pipeline([
        ("vec", criar_vectorizador(rep, usar_stop, ngram)),
        ("clf", criar_classificador(clf, alpha, C)),
    ])


def prever(pipe: Pipeline, texto: str):
    """Devolve (vector x, probabilidades, classes, tempo em ms)."""
    vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]
    x = vec.transform([texto])
    t0 = time.perf_counter()
    proba = clf.predict_proba(x)[0]
    ms = (time.perf_counter() - t0) * 1000
    return x, proba, clf.classes_, ms


def _pesos_por_classe(clf):
    """Matriz (classes × termos) cujas diferenças entre linhas são log-odds.

    NB: log P(termo | classe).  RL multinomial: coeficientes da softmax.
    Em ambos, x_j · (w[c1, j] − w[c2, j]) é quanto o termo j empurra a decisão de c2 para c1.
    """
    if isinstance(clf, MultinomialNB):
        return clf.feature_log_prob_
    coef = clf.coef_
    if coef.shape[0] == 1:  # caso binário: sklearn guarda só uma linha
        coef = np.vstack([-coef[0] / 2, coef[0] / 2])
    return coef


def contribuicoes(pipe: Pipeline, texto: str, x, proba, classes, max_termos: int = 15):
    """Contribuição de cada termo para 'classe vencedora vs. segunda classe'.

    Devolve (DataFrame termo/contribuição, termos desconhecidos, classe 1, classe 2).
    Com menos de 2 classes não há comparação possível → DataFrame vazio.
    """
    vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]
    termos = list(dict.fromkeys(vec.build_analyzer()(texto)))
    conhecidos = [t for t in termos if t in vec.vocabulary_]
    desconhecidos = [t for t in termos if t not in vec.vocabulary_]
    if len(classes) < 2:
        return pd.DataFrame(columns=["termo", "contribuição"]), desconhecidos, classes[0], None

    ordem = np.argsort(proba)[::-1]
    i1, i2 = ordem[0], ordem[1]
    w = _pesos_por_classe(clf)
    linhas = []
    for t in conhecidos:
        j = vec.vocabulary_[t]
        linhas.append({"termo": t, "contribuição": float(x[0, j] * (w[i1, j] - w[i2, j]))})
    d = pd.DataFrame(linhas, columns=["termo", "contribuição"])
    if len(d) > max_termos:  # n-gramas de caracteres geram muitos termos: mostramos os mais fortes
        d = d.reindex(d["contribuição"].abs().sort_values(ascending=False).index).head(max_termos)
    return d.reset_index(drop=True), desconhecidos, classes[i1], classes[i2]
