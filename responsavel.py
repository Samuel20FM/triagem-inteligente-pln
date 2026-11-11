"""IA responsável (tema 15): privacidade (anonimização) e viés (métricas por subgrupo, reponderação)."""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Ordem importa: padrões mais específicos primeiro.
PADROES_PESSOAIS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    # BI moçambicano: 12 dígitos + 1 letra
    ("BI", re.compile(r"(?<!\w)\d{12}[A-Za-z](?!\w)")),
    # Telemóvel moçambicano: 82–87 + 7 dígitos, com ou sem +258 e espaços/hífens
    ("TELEFONE", re.compile(r"(?<!\d)(?:\+?258[\s-]?)?8[2-7](?:[\s-]?\d){7}(?!\d)")),
    # Qualquer outro número longo (NUIT, nº de estudante, conta…) pode identificar alguém
    ("NUMERO_ID", re.compile(r"(?<!\d)\d{6,11}(?!\d)")),
]


def anonimizar(texto: str):
    """Substitui dados pessoais por marcadores, ex.: '84 123 4567' → '[TELEFONE]'.

    Devolve (texto mascarado, lista de (tipo, valor encontrado)).
    Limitação: nomes próprios não são detectados (precisaria de um modelo de entidades).
    """
    encontrados = []
    for tipo, padrao in PADROES_PESSOAIS:
        def _troca(m, tipo=tipo):
            encontrados.append((tipo, m.group(0)))
            return f"[{tipo}]"
        texto = padrao.sub(_troca, texto)
    return texto, encontrados


def metricas_por_subgrupo(teste: pd.DataFrame, pred, coluna: str = "registo") -> pd.DataFrame:
    """Exatidão e F1 macro da categoria em cada subgrupo (ex.: formal vs informal)."""
    d = teste.assign(_pred=np.asarray(pred))
    linhas = []
    for g, parte in d.groupby(coluna):
        linhas.append({coluna: g, "n": len(parte),
                       "exatidão": accuracy_score(parte["categoria"], parte["_pred"]),
                       "F1 macro": f1_score(parte["categoria"], parte["_pred"], average="macro", zero_division=0)})
    return pd.DataFrame(linhas)


def metricas_urgencia_por_subgrupo(teste: pd.DataFrame, p_urg, limiar: float,
                                   coluna: str = "registo") -> pd.DataFrame:
    """Para cada subgrupo: taxa real de urgentes, taxa prevista, revocação (TPR) e falsos alarmes (FPR).

    Igualdade de oportunidades: a revocação dos urgentes deve ser parecida entre subgrupos.
    """
    d = teste.assign(_p=np.asarray(p_urg), _pred=(np.asarray(p_urg) >= limiar).astype(int))
    linhas = []
    for g, parte in d.groupby(coluna):
        y, pr = parte["urgente"].astype(int), parte["_pred"]
        pos, neg = (y == 1), (y == 0)
        linhas.append({coluna: g, "n": len(parte),
                       "urgentes (rótulo)": y.mean(),
                       "urgentes (previsto)": pr.mean(),
                       "revocação (TPR)": (pr[pos] == 1).mean() if pos.any() else np.nan,
                       "falsos alarmes (FPR)": (pr[neg] == 1).mean() if neg.any() else np.nan})
    return pd.DataFrame(linhas)


def taxa_urgencia_a_categoria_igual(df: pd.DataFrame, coluna: str = "registo") -> pd.DataFrame:
    """Taxa de urgência no RÓTULO HISTÓRICO por categoria × subgrupo.

    Se, dentro da mesma categoria, um subgrupo é marcado urgente muito mais vezes,
    o rótulo pode reflectir a forma de escrever e não a urgência do pedido.
    """
    return (df.groupby(["categoria", coluna])["urgente"].mean().unstack(coluna))


def limiares_por_subgrupo(grupo, y, p, revocacao_alvo: float = 0.7, limiar_max: float = 0.5) -> dict:
    """Pós-processamento (igualdade de oportunidades, Hardt et al., 2016).

    Para cada subgrupo escolhe o MAIOR limiar ≤ limiar_max que ainda apanha pelo menos
    `revocacao_alvo` dos urgentes. Deve ser calculado com previsões fora-da-amostra do treino.
    """
    grupo, y, p = np.asarray(grupo), np.asarray(y).astype(int), np.asarray(p)
    limiares = {}
    for g in np.unique(grupo):
        s = (grupo == g) & (y == 1)
        cand = [l for l in np.round(np.arange(0.05, limiar_max + 1e-9, 0.01), 2)
                if s.any() and (p[s] >= l).mean() >= revocacao_alvo]
        limiares[g] = float(max(cand)) if cand else 0.05
    return limiares


def pesos_reponderacao(grupo, y) -> np.ndarray:
    """Reweighing (Kamiran & Calders, 2012): w(g, y) = P(g)·P(y) / P(g, y).

    Dá mais peso às combinações sub-representadas (ex.: informal & urgente) para que,
    no treino, o subgrupo deixe de prever o rótulo.
    """
    d = pd.DataFrame({"g": np.asarray(grupo), "y": np.asarray(y)})
    n = len(d)
    p_g = d["g"].map(d["g"].value_counts() / n)
    p_y = d["y"].map(d["y"].value_counts() / n)
    p_gy = d.groupby(["g", "y"])["y"].transform("size") / n
    return (p_g * p_y / p_gy).to_numpy()
