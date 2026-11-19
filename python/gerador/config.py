"""Leitura do arquivo de configuração do gerador."""
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_PADRAO = RAIZ / "config" / "config.yaml"


def carregar_config(caminho: Path | str = CAMINHO_PADRAO) -> dict:
    """Lê o YAML de configuração e devolve um dicionário."""
    with open(caminho, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)