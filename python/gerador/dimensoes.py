"""Gera as tabelas de apoio sintéticas: polos, LEADTIME, prazos de coleta especiais, feriados e ocorrências."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .calendario import tabela_feriados

# Ocorrências que IMPÕEM um status: (ocorrência, status de coleta, status de embarque, status de entrega).
# O vocabulário é genérico: não cita cliente nem transportadora reais.
_OCORRENCIAS_COM_STATUS = [
    ("COLETADO - TRANSP. ALTERNATIVA", "COLETADO PELA TRANSP. ALTERNATIVA",
     "EMBARCADO PELA TRANSP. ALTERNATIVA", "ENTREGUE PELA TRANSP. ALTERNATIVA"),
    ("COLETA CANCELADA DEVIDO A INVENTÁRIO", "INVENTÁRIO", "INVENTÁRIO", "INVENTÁRIO"),
    ("ATRASO DEVIDO AO POLO", "COLETA ATRASADA PELO POLO",
     "EMBARQUE ATRASADO PELO POLO", "ENTREGA ATRASADA PELO POLO"),
    ("OM CANCELADA", "CANCELADO PELO CLIENTE", "CANCELADO PELO CLIENTE", "CANCELADO PELO CLIENTE"),
]

# Ocorrências que NÃO alteram o status calculado.
_OCORRENCIAS_SEM_STATUS = [
    "AG POSICIONAMENTO DO CLIENTE", "AG CLIENTE VIRAR OM", "AGUARDANDO CONFIRMAÇÃO",
    "AGUARDANDO CUBAGEM", "AGUARDANDO EMBARQUE", "AGUARDANDO FOTO",
    "COLETA FORA DA SOLICITAÇÃO", "COLETADO SEM FOTO", "OM DE COTAÇÃO", "OM FRUSTRADA",
    "ATRASO DEVIDO AO PARCEIRO", "OM NÃO DISPONÍVEL", "OM NÃO DISPONÍVEL PARA COLETA NO DIA",
    "REAGENDADO A COLETA", "SINISTRO", "OM COLETADA PELO CLIENTE",
    "OM COLETADA POR OUTRA TRANSPORTADORA",
]


def carregar_municipios(caminho: Path | str) -> pd.DataFrame:
    """Lê a lista pública de municípios (colunas UF e CIDADE) e padroniza o texto."""
    municipios = pd.read_excel(caminho, dtype=str)
    municipios.columns = [str(coluna).strip().upper() for coluna in municipios.columns]
    municipios = municipios[["UF", "CIDADE"]].dropna()
    municipios["UF"] = municipios["UF"].str.strip().str.upper()
    municipios["CIDADE"] = municipios["CIDADE"].str.strip().str.upper()
    return municipios.drop_duplicates().reset_index(drop=True)


def escolher_cidades_atendidas(cfg: dict, rng: np.random.Generator, municipios: pd.DataFrame) -> pd.DataFrame:
    """Sorteia as UFs e as cidades onde a operação fictícia coleta."""
    estrutura = cfg["estrutura"]
    total_ufs = estrutura["quantidade_ufs"]
    total_cidades = estrutura["cidades_atendidas"]

    contagem = municipios["UF"].value_counts()
    ufs_possiveis = sorted(contagem[contagem >= 10].index)  # evita UFs com poucos municípios
    ufs = sorted(rng.choice(ufs_possiveis, size=total_ufs, replace=False))

    por_uf = np.full(total_ufs, 3)  # cada UF recebe pelo menos 3 cidades
    sobra = total_cidades - int(por_uf.sum())
    por_uf = por_uf + rng.multinomial(sobra, rng.dirichlet(np.ones(total_ufs)))

    partes = []
    for uf, quantidade in zip(ufs, por_uf):
        cidades_da_uf = municipios.loc[municipios["UF"] == uf, "CIDADE"].to_numpy()
        quantidade = min(int(quantidade), len(cidades_da_uf))
        sorteadas = rng.choice(cidades_da_uf, size=quantidade, replace=False)
        partes.append(pd.DataFrame({"UF": uf, "CIDADE": sorteadas}))
    return pd.concat(partes, ignore_index=True)


def montar_polos(cfg: dict, rng: np.random.Generator, cidades: pd.DataFrame) -> pd.DataFrame:
    """Liga cada cidade a um polo, um representante e um modal (todos fictícios)."""
    estrutura = cfg["estrutura"]
    total_polos = estrutura["quantidade_polos"]
    total_reps = estrutura["quantidade_representantes"]
    modais = estrutura["modais"]

    codigos_polo = [f"{estrutura['prefixo_polo']}{i:02d}" for i in range(1, total_polos + 1)]
    codigos_rep = [f"{estrutura['prefixo_representante']}{i:02d}" for i in range(1, total_reps + 1)]

    ufs = sorted(cidades["UF"].unique())
    polos_da_uf = {uf: [codigos_polo[i % total_polos]] for i, uf in enumerate(ufs)}
    # os polos que sobram (além de um por UF) reforçam as UFs com mais cidades
    ufs_maiores = list(cidades["UF"].value_counts().index[: max(0, total_polos - len(ufs))])
    for uf, polo_extra in zip(ufs_maiores, codigos_polo[len(ufs):]):
        polos_da_uf[uf].append(polo_extra)

    reps_do_polo = {
        polo: [str(r) for r in rng.choice(codigos_rep, size=int(rng.integers(1, 3)), replace=False)]
        for polo in codigos_polo
    }
    pesos = np.linspace(len(modais), 1, len(modais))
    pesos = pesos / pesos.sum()  # o primeiro modal é o mais usado

    linhas = []
    for uf, cidade in zip(cidades["UF"], cidades["CIDADE"]):
        polo = str(rng.choice(polos_da_uf[uf]))
        linhas.append({
            "ORIGEM": cidade,
            "UF": uf,
            "SIGLA": polo,
            "REPRESENTANTE": str(rng.choice(reps_do_polo[polo])),
            "MODAL": str(rng.choice(modais, p=pesos)),
        })
    return pd.DataFrame(linhas)


def gerar_leadtime(cfg: dict, rng: np.random.Generator, polos: pd.DataFrame) -> pd.DataFrame:
    """Cria o prazo de trânsito (dias úteis) de cada cidade: base da UF mais uma pequena variação."""
    limites = cfg["prazos"]["leadtime"]
    minimo, maximo = limites["dias_minimo"], limites["dias_maximo"]
    base_da_uf = {uf: int(rng.integers(minimo + 1, maximo - 1)) for uf in sorted(polos["UF"].unique())}

    dias = []
    for uf in polos["UF"]:
        valor = base_da_uf[uf] + int(rng.integers(-1, 3))
        dias.append(int(np.clip(valor, minimo, maximo)))
    return pd.DataFrame({
        "MUNICIPIO": polos["ORIGEM"],
        "UF": polos["UF"],
        "TRANSIT_DIAS_UTEIS": dias,
        "OBSERVACOES": "",
    })


def escolher_prazos_especiais(cfg: dict, rng: np.random.Generator, leadtime: pd.DataFrame) -> pd.DataFrame:
    """Escolhe, entre as cidades mais distantes, as que têm prazo de coleta maior que o padrão."""
    especial = cfg["prazos"]["coleta_especial"]
    padrao = cfg["prazos"]["coleta_padrao_dias_uteis"]
    quantidade = especial["quantidade_cidades"]

    candidatas = leadtime.sort_values("TRANSIT_DIAS_UTEIS", ascending=False).head(quantidade * 3)
    posicoes = rng.choice(len(candidatas), size=quantidade, replace=False)
    escolhidas = candidatas.iloc[posicoes]
    prazos = rng.integers(especial["dias_minimo"], especial["dias_maximo"] + 1, size=quantidade)
    return pd.DataFrame({
        "CIDADE": escolhidas["MUNICIPIO"].to_numpy(),
        "UF": escolhidas["UF"].to_numpy(),
        "PRAZO": prazos,
        "MOTIVO": f"Distância e dificuldade de coleta em {padrao} dias úteis",
    })


def gerar_ocorrencias() -> pd.DataFrame:
    """Lista de ocorrências e os status que algumas impõem (as demais deixam os campos vazios)."""
    colunas = ["OCORRÊNCIA", "STATUS COLETA", "STATUS EMBARQUE", "STATUS ENTREGA"]
    com_status = pd.DataFrame(_OCORRENCIAS_COM_STATUS, columns=colunas)
    sem_status = pd.DataFrame({"OCORRÊNCIA": _OCORRENCIAS_SEM_STATUS})
    sem_status[colunas[1:]] = ""
    return pd.concat([com_status, sem_status], ignore_index=True)


def gerar_dimensoes(cfg: dict, municipios: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Gera todas as tabelas de apoio. Com a mesma semente, o resultado é sempre o mesmo."""
    rng = np.random.default_rng(cfg["semente"])
    cidades = escolher_cidades_atendidas(cfg, rng, municipios)
    polos = montar_polos(cfg, rng, cidades)
    leadtime = gerar_leadtime(cfg, rng, polos)
    prazos_especiais = escolher_prazos_especiais(cfg, rng, leadtime)
    return {
        "polos": polos,
        "leadtime": leadtime,
        "prazos_coleta_especial": prazos_especiais,
        "feriados": tabela_feriados(),
        "ocorrencias": gerar_ocorrencias(),
    }