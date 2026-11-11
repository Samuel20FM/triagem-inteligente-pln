import numpy as np
import pytest

from representacoes import REPRESENTACOES, criar_vectorizador, ngramas_partilhados


def test_bow_gera_contagens_inteiras():
    X = criar_vectorizador("bow").fit_transform(["propina propina paga"])
    assert X.max() == 2


@pytest.mark.parametrize("chave", ["tfidf", "char"])
def test_tfidf_vectores_normalizados(chave):
    X = criar_vectorizador(chave).fit_transform(["propina nao validada", "pagar propina", "projector avariado"])
    assert np.allclose(np.sqrt(X.multiply(X).sum(axis=1)).A1, 1.0)


def test_char_resiste_a_gralhas():
    assert ngramas_partilhados("matricula", "matricla") >= 0.35
    assert ngramas_partilhados("matricula", "projector") < 0.05


def test_char_ve_gralha_que_palavras_nao_veem():
    textos = ["confirmar a matricula", "projector avariado"]
    vw, vc = criar_vectorizador("tfidf").fit(textos), criar_vectorizador("char").fit(textos)
    assert vw.transform(["matricla"]).nnz == 0       # palavra desconhecida → vector vazio
    assert vc.transform(["matricla"]).nnz > 0        # pedaços conhecidos → ainda há sinal


def test_representacao_desconhecida():
    with pytest.raises(ValueError):
        criar_vectorizador("xpto")


def test_tres_representacoes():
    assert set(REPRESENTACOES) == {"bow", "tfidf", "char"}
