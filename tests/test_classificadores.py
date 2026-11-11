import numpy as np
import pytest

from classificadores import contribuicoes, criar_pipeline, prever


@pytest.mark.parametrize("chave", [("bow", "nb"), ("tfidf", "nb"), ("char", "nb"),
                                   ("bow", "lr"), ("tfidf", "lr"), ("char", "lr")])
def test_seis_combinacoes_probabilidades_somam_um(R, chave):
    _, proba, _, _ = prever(R["modelos"][chave]["pipe"], "qualquer coisa")
    assert pytest.approx(proba.sum()) == 1.0


@pytest.mark.parametrize("clf", ["nb", "lr"])
@pytest.mark.parametrize("texto,esperada", [
    ("o pagamento da propina nao foi validado", "financeiro"),
    ("o projector da sala 14 esta avariado", "infraestrutura"),
    ("quando sai a pauta de analise matematica", "notas"),
    ("quero confirmar a minha inscricao", "matricula"),
])
def test_previsoes_tipicas(R, clf, texto, esperada):
    _, proba, cls, _ = prever(R["modelos"][("tfidf", clf)]["pipe"], texto)
    assert cls[np.argmax(proba)] == esperada


@pytest.mark.parametrize("clf", ["nb", "lr"])
def test_contribuicoes_apontam_para_vencedora(R, clf):
    pipe = R["modelos"][("tfidf", clf)]["pipe"]
    texto = "o pagamento da propina nao foi validado"
    x, proba, cls, _ = prever(pipe, texto)
    d, _, c1, _ = contribuicoes(pipe, texto, x, proba, cls)
    assert c1 == "financeiro" and d["contribuição"].sum() > 0


def test_palavras_desconhecidas_nao_quebram(R):
    pipe = R["modelos"][("tfidf", "nb")]["pipe"]
    x, proba, cls, _ = prever(pipe, "xyzzy qwerty")
    d, desc, _, _ = contribuicoes(pipe, "xyzzy qwerty", x, proba, cls)
    assert d.empty and set(desc) == {"xyzzy", "qwerty"}


def test_contribuicoes_com_uma_classe_nao_rebenta():
    pipe = criar_pipeline("tfidf", "nb").fit(["pauta de notas", "nota do exame"], ["notas", "notas"])
    x, proba, cls, _ = prever(pipe, "pauta")
    d, _, _, c2 = contribuicoes(pipe, "pauta", x, proba, cls)
    assert d.empty and c2 is None


def test_acento_e_maiuscula_dao_mesma_previsao(R):
    p = R["modelos"][("char", "lr")]["pipe"]
    assert np.allclose(prever(p, "Não consigo pagar a PROPINA")[1], prever(p, "nao consigo pagar a propina")[1])
