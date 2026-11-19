import pandas as pd
import pytest

from gerador.calendario import dias_uteis_entre, somar_dias_uteis, tabela_feriados

FERIADOS = tabela_feriados()["DATA"]


@pytest.mark.parametrize(
    "inicio, dias, esperado",
    [
        ("2026-01-05", 2, "2026-01-07"),
        ("2026-01-16", 3, "2026-01-21"),
        ("2026-01-08", 3, "2026-01-13"),
        ("2026-01-21", 3, "2026-01-26"),
        ("2026-01-05", 5, "2026-01-12"),
    ],
)
def test_somar_dias_uteis_igual_ao_excel(inicio, dias, esperado):
    assert somar_dias_uteis(inicio, dias, FERIADOS) == pd.Timestamp(esperado)


def test_dias_uteis_entre_inclui_as_duas_pontas():
    # de segunda a sexta: 5 dias úteis
    assert dias_uteis_entre("2026-01-05", "2026-01-09", FERIADOS) == 5


def test_dias_uteis_entre_desconta_o_carnaval():
    # 02/02 a 20/02 tem 15 dias úteis; descontando 16 e 17/02 (Carnaval), ficam 13
    assert dias_uteis_entre("2026-02-02", "2026-02-20", FERIADOS) == 13


def test_dias_uteis_entre_devolve_negativo_quando_invertido():
    assert dias_uteis_entre("2026-01-09", "2026-01-05", FERIADOS) == -5