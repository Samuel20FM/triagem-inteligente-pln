"""Representações de texto (tema 13): Bag-of-Words, TF-IDF de palavras e TF-IDF de n-gramas de caracteres."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from texto import STOPWORDS_PT, normalizar

REPRESENTACOES = {
    "bow": "Bag-of-Words",
    "tfidf": "TF-IDF (palavras)",
    "char": "TF-IDF (caracteres)",
}

# Tamanho dos pedaços de letras usados na representação "char".
NGRAMA_CHAR = (3, 5)


def criar_vectorizador(chave: str, usar_stop: bool = True, ngram: tuple = (1, 1)):
    """Devolve o vectorizador (ainda por treinar) da representação pedida.

    - bow:   conta palavras (inteiros).
    - tfidf: TF × IDF por palavra, normalizado (norma L2 = 1).
    - char:  TF-IDF sobre pedaços de 3 a 5 letras dentro de cada palavra ("char_wb").
             Stop-words e bigramas de palavras não se aplicam aqui.
    """
    stop = STOPWORDS_PT if usar_stop else None
    if chave == "bow":
        return CountVectorizer(preprocessor=normalizar, stop_words=stop, ngram_range=ngram)
    if chave == "tfidf":
        return TfidfVectorizer(preprocessor=normalizar, stop_words=stop, ngram_range=ngram)
    if chave == "char":
        return TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb", ngram_range=NGRAMA_CHAR)
    raise ValueError(f"Representação desconhecida: {chave}")


def ngramas_caracteres(palavra: str) -> set:
    """Pedaços de 3 a 5 letras de uma palavra, como o vectorizador 'char' os vê."""
    return set(criar_vectorizador("char").build_analyzer()(palavra))


def ngramas_partilhados(a: str, b: str) -> float:
    """Fracção de n-gramas em comum (Jaccard). Mostra porque 'matricla' ≈ 'matricula'."""
    na, nb = ngramas_caracteres(a), ngramas_caracteres(b)
    if not na and not nb:
        return 0.0
    return len(na & nb) / len(na | nb)
