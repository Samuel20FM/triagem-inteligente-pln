"""Pré-processamento de texto (tema 13 — preparação do texto)."""
import re
import unicodedata

# Stop-words SEM acentos (o texto é normalizado antes). A negação
# ("nao", "sem", "nem") é mantida de propósito: muda o sentido da frase.
STOPWORDS_PT = [
    "a", "o", "as", "os", "de", "do", "da", "dos", "das", "em", "no", "na", "nos",
    "nas", "por", "para", "com", "um", "uma", "uns", "umas", "ao", "aos", "e", "ou",
    "que", "se", "me", "meu", "minha", "meus", "minhas", "este", "esta", "isto",
    "isso", "pelo", "pela",
]


def normalizar(texto: str) -> str:
    """minúsculas, sem acentos, sem pontuação, espaços únicos."""
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
