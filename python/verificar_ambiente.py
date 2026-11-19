"""Verifica o ambiente e reproduz em Python os testes de aceitação do Excel (Módulo 1)."""

import sys

import numpy as np
import pandas as pd

# Feriados de 2026 usados nos testes (os mesmos da tabela de feriados do Excel).
FERIADOS_2026 = np.array(
    [
        "2026-01-01",
        "2026-01-25",
        "2026-02-16",
        "2026-02-17",
        "2026-04-03",
        "2026-04-21",
        "2026-05-01",
        "2026-06-04",
        "2026-07-09",
        "2026-09-07",
        "2026-10-12",
        "2026-11-02",
        "2026-11-20",
        "2026-12-25",
        "2026-12-31",
    ],
    dtype="datetime64[D]",
)


def somar_dias_uteis(data_inicial: str, dias: int) -> str:
    """Soma dias úteis (sem sábado, domingo e feriados), como o DIATRABALHO do Excel."""
    resultado = np.busday_offset(
        np.datetime64(data_inicial),
        dias,
        roll="backward",
        holidays=FERIADOS_2026,
    )
    return str(resultado)


TESTES = [
    ("ARACRUZ", "2026-01-05", 2, "2026-01-07"),
    ("CHAPADAO DO SUL", "2026-01-16", 3, "2026-01-21"),
    ("COXIM", "2026-01-08", 3, "2026-01-13"),
    ("CORUMBA", "2026-01-21", 3, "2026-01-26"),
    ("BARRA DE SAO FRANCISCO", "2026-01-05", 5, "2026-01-12"),
]


def main() -> None:
    print(
        f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}"
    )

    falhas = 0

    for cidade, inicio, prazo, esperado in TESTES:
        obtido = somar_dias_uteis(inicio, prazo)
        status = "OK" if obtido == esperado else "REVISAR"
        falhas += status != "OK"

        print(
            f"{cidade:<26} {inicio} + {prazo} dias úteis = "
            f"{obtido} (esperado {esperado}) {status}"
        )

    print(
        "Todos os testes passaram."
        if falhas == 0
        else f"{falhas} teste(s) falharam."
    )


if __name__ == "__main__":
    main()