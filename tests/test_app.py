"""Testes da interface com o AppTest do Streamlit."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def at():
    return AppTest.from_file(APP, default_timeout=240).run()


def test_app_arranca_com_sete_separadores(at):
    assert not at.exception
    assert len(at.tabs) == 7


def test_app_classifica_mensagem(at):
    at.text_area[0].set_value("o projector da sala 14 nao funciona").run()
    assert not at.exception
    assert "Infraestrutura" in " ".join(m.value for m in at.markdown)


def test_app_mensagem_vazia_mostra_aviso(at):
    at.text_area[0].set_value("   ").run()
    assert any("Escreva uma mensagem" in i.value for i in at.info)


def test_app_mascara_telefone(at):
    at.text_area[0].set_value("o projector avariou, liguem 84 123 4567").run()
    assert not at.exception
    assert any("[TELEFONE]" in i.value for i in at.info)


@pytest.mark.parametrize("accao", ["bigramas", "stop", "teste33"])
def test_app_parametros(at, accao):
    if accao == "bigramas":
        at.sidebar.radio[0].set_value("Palavras + pares (bigramas)")
    elif accao == "stop":
        at.sidebar.checkbox[0].uncheck()
    else:
        at.sidebar.select_slider[0].set_value(0.33)
    at.run()
    assert not at.exception
