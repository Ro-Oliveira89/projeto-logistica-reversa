"""Calendário de dias úteis: o equivalente em Python de DIATRABALHO e DIATRABALHOTOTAL do Excel."""
from __future__ import annotations

import numpy as np
import pandas as pd

# (data, feriado, âmbito). As datas são públicas; o âmbito segue a classificação do projeto.
_FERIADOS = [
    ("2026-01-01", "Confraternização Universal", "Nacional"),
    ("2026-01-25", "Aniversário de São Paulo", "Municipal (SP)"),
    ("2026-02-16", "Carnaval", "Ponto facultativo"),
    ("2026-02-17", "Carnaval", "Ponto facultativo"),
    ("2026-04-03", "Paixão de Cristo", "Nacional"),
    ("2026-04-21", "Tiradentes", "Nacional"),
    ("2026-05-01", "Dia do Trabalho", "Nacional"),
    ("2026-06-04", "Corpus Christi", "Ponto facultativo"),
    ("2026-07-09", "Revolução de 1932", "Estadual (SP)"),
    ("2026-09-07", "Independência do Brasil", "Nacional"),
    ("2026-10-12", "Nossa Sra. Aparecida", "Nacional"),
    ("2026-11-02", "Finados", "Nacional"),
    ("2026-11-15", "Proclamação da República", "Nacional"),
    ("2026-11-20", "Consciência Negra", "Nacional"),
    ("2026-12-25", "Natal", "Nacional"),
    ("2026-12-31", "Réveilon", "Interno"),
    ("2027-01-01", "Confraternização Universal", "Nacional"),
]


def tabela_feriados() -> pd.DataFrame:
    """Devolve a tabela de feriados com as colunas DATA, FERIADO e AMBITO."""
    tabela = pd.DataFrame(_FERIADOS, columns=["DATA", "FERIADO", "AMBITO"])
    tabela["DATA"] = pd.to_datetime(tabela["DATA"])
    return tabela


def _como_array(feriados) -> np.ndarray:
    """Converte uma lista de datas para o formato que o numpy usa nas contas de dias úteis."""
    return pd.to_datetime(feriados).values.astype("datetime64[D]")


def somar_dias_uteis(data, dias: int, feriados) -> pd.Timestamp:
    """Soma dias úteis a uma data (sem sábado, domingo e feriados). Equivale ao DIATRABALHO."""
    inicio = np.datetime64(pd.Timestamp(data).date())
    resultado = np.busday_offset(inicio, dias, roll="backward", holidays=_como_array(feriados))
    return pd.Timestamp(resultado)


def dias_uteis_entre(inicio, fim, feriados) -> int:
    """Conta os dias úteis entre duas datas, incluindo as duas pontas (DIATRABALHOTOTAL).

    Devolve um número negativo quando a data final é anterior à inicial, como o Excel.
    """
    a = np.datetime64(pd.Timestamp(inicio).date())
    b = np.datetime64(pd.Timestamp(fim).date())
    h = _como_array(feriados)
    if b >= a:
        return int(np.busday_count(a, b + 1, holidays=h))
    return -int(np.busday_count(b, a + 1, holidays=h))