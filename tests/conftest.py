"""Configuração comum dos testes. Executar a partir da pasta triagem/:  pytest -v"""
import logging
import sys
import warnings
from pathlib import Path

import pytest

PASTA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PASTA))
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from avaliacao import ler_csv, treinar  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return ler_csv(PASTA / "corpus_referencia.csv")


@pytest.fixture(scope="session")
def R(df):
    return treinar(df, proporcao_teste=0.2, seed=42)
