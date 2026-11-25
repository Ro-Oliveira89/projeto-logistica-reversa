"""Gera as OMs sintéticas e simula o caminho de cada uma: coleta, embarque e entrega."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendario import UM_DIA, tabela_feriados

COLUNAS_CONTROLE = [
    "OM", "ORIGEM", "UF", "QTD", "DATA DISP COLETA", "DATA SOLICITAÇÃO",
    "DATA EFETIVA DA COLETA", "DACTE", "OST", "MINUTA DE EMBARQUE", "DATA DO EMBARQUE",
    "DATA EFETIVA ENTREGA", "OCORRÊNCIA", "DATA OCORRÊNCIA", "OBSERVAÇÕES",
]

# Ocorrências sorteadas para as OMs (nome, probabilidade). As demais OMs ficam sem ocorrência.
_OCORRENCIAS_SORTEADAS = [
    ("COLETADO - TRANSP. ALTERNATIVA", 0.055),
    ("COLETA FORA DA SOLICITAÇÃO", 0.060),
    ("SINISTRO", 0.008),
    ("REAGENDADO A COLETA", 0.005),
    ("OM FRUSTRADA", 0.003),
    ("OM NÃO DISPONÍVEL", 0.003),
    ("OM COLETADA POR OUTRA TRANSPORTADORA", 0.002),
    ("COLETA CANCELADA DEVIDO A INVENTÁRIO", 0.002),
    ("OM CANCELADA", 0.003),
]
_OBSERVACOES = [
    "Cliente pediu reagendamento", "Aguardando liberação da carga",
    "Contato realizado com a franquia", "Coleta feita no período da tarde",
    "Carga conferida na base",
]


def _array_feriados(feriados: pd.Series) -> np.ndarray:
    return pd.to_datetime(feriados).values.astype("datetime64[D]")


def dias_uteis_do_periodo(cfg: dict, feriados: np.ndarray) -> np.ndarray:
    """Todos os dias úteis entre o início e o fim do período configurado."""
    inicio = np.datetime64(cfg["periodo"]["inicio"], "D")
    fim = np.datetime64(cfg["periodo"]["fim"], "D")
    dias = np.arange(inicio, fim + UM_DIA, dtype="datetime64[D]")
    return dias[np.is_busday(dias, holidays=feriados)]


def sortear_solicitacoes(cfg: dict, rng: np.random.Generator, dias: np.ndarray) -> np.ndarray:
    """Sorteia a data de solicitação de cada OM, com meses mais e menos movimentados."""
    meses = dias.astype("datetime64[M]")
    fator_do_mes = {mes: rng.uniform(0.5, 1.5) for mes in np.unique(meses)}
    pesos = np.array([fator_do_mes[mes] for mes in meses])
    posicoes = rng.choice(len(dias), size=cfg["volume"]["total_oms"], p=pesos / pesos.sum())
    return np.sort(dias[posicoes])


def montar_perfis_dos_polos(rng: np.random.Generator, volume_por_polo: pd.Series) -> pd.DataFrame:
    """Cada polo tem a sua chance de atrasar a coleta e o seu atraso médio no embarque.

    Um polo (entre os de maior volume) é o 'polo problema': coleta bem, mas embarca devagar.
    Outro é o 'polo exemplar'. Os números são inventados, sem relação com a operação real.
    """
    polos = list(volume_por_polo.index)
    perfis = pd.DataFrame({
        "SIGLA": polos,
        "CHANCE_ATRASO_COLETA": rng.uniform(0.10, 0.30, size=len(polos)),
        "ATRASO_MEDIO_EMBARQUE": rng.uniform(1.2, 2.6, size=len(polos)),
    }).set_index("SIGLA")
    maiores = list(volume_por_polo.sort_values(ascending=False).index[:3])
    polo_problema = str(rng.choice(maiores))
    menores = [p for p in polos if p not in maiores]  # o exemplar não domina o volume total
    polo_exemplar = str(rng.choice(menores))
    perfis.loc[polo_problema, ["CHANCE_ATRASO_COLETA", "ATRASO_MEDIO_EMBARQUE"]] = [0.12, 4.8]
    perfis.loc[polo_exemplar, ["CHANCE_ATRASO_COLETA", "ATRASO_MEDIO_EMBARQUE"]] = [0.05, 0.8]
    perfis["PERFIL"] = "tipico"
    perfis.loc[polo_problema, "PERFIL"] = "problema"
    perfis.loc[polo_exemplar, "PERFIL"] = "exemplar"
    return perfis


def _somar_uteis(inicio: np.ndarray, dias: np.ndarray, feriados: np.ndarray) -> np.ndarray:
    return np.busday_offset(inicio, dias, roll="backward", holidays=feriados)


def gerar_oms(cfg: dict, tabelas: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cria as OMs e simula a coleta, o embarque e a entrega de cada uma.

    Devolve (oms, perfis_dos_polos). A tabela 'oms' tem as colunas do CONTROLE e algumas
    colunas de apoio (SIGLA, LEADTIME, PRAZO_COLETA) usadas nas conferências.
    """
    rng = np.random.default_rng(cfg["semente"] + 1)  # sequência própria, independente das tabelas de apoio
    feriados = _array_feriados(tabelas["feriados"]["DATA"])
    total = cfg["volume"]["total_oms"]
    corte = np.datetime64(cfg["periodo"]["fim"], "D")
    padrao = cfg["prazos"]["coleta_padrao_dias_uteis"]

    # 1. Solicitação, origem e quantidade
    solicitacao = sortear_solicitacoes(cfg, rng, dias_uteis_do_periodo(cfg, feriados))
    polos = tabelas["polos"].merge(
        tabelas["leadtime"][["MUNICIPIO", "UF", "TRANSIT_DIAS_UTEIS"]],
        left_on=["ORIGEM", "UF"], right_on=["MUNICIPIO", "UF"],
    ).drop(columns="MUNICIPIO")
    pesos_cidade = rng.pareto(1.3, size=len(polos)) + 1  # poucas cidades concentram muitas OMs
    cidades = rng.choice(len(polos), size=total, p=pesos_cidade / pesos_cidade.sum())
    oms = polos.iloc[cidades].reset_index(drop=True)
    oms = oms.rename(columns={"TRANSIT_DIAS_UTEIS": "LEADTIME"})

    especiais = {(c, u): p for c, u, p in zip(
        tabelas["prazos_coleta_especial"]["CIDADE"],
        tabelas["prazos_coleta_especial"]["UF"],
        tabelas["prazos_coleta_especial"]["PRAZO"])}
    oms["PRAZO_COLETA"] = [especiais.get((c, u), padrao) for c, u in zip(oms["ORIGEM"], oms["UF"])]
    oms["QTD"] = rng.poisson(8, size=total) + 1
    atraso_disp = np.minimum(np.floor(rng.exponential(2.0, size=total)), 25).astype(int)
    disp = solicitacao - atraso_disp.astype("timedelta64[D]")

    # 2. Perfil de cada polo
    perfis = montar_perfis_dos_polos(rng, oms["SIGLA"].value_counts())
    chance_atraso = oms["SIGLA"].map(perfis["CHANCE_ATRASO_COLETA"]).to_numpy()
    atraso_embarque = oms["SIGLA"].map(perfis["ATRASO_MEDIO_EMBARQUE"]).to_numpy()
    prazo = oms["PRAZO_COLETA"].to_numpy()
    leadtime = oms["LEADTIME"].to_numpy()

    # 3. Caminho de cada OM, contado em dias úteis a partir da solicitação
    coleta_atrasada = rng.random(total) < chance_atraso
    dias_coleta = np.where(
        coleta_atrasada,
        prazo + rng.geometric(0.45, size=total),            # atrasa de 1 dia útil em diante
        (rng.random(total) * (prazo + 1)).astype(int),      # no prazo: de 0 até o prazo
    )
    p_embarque = 1 / (1 + atraso_embarque)
    dias_embarque = dias_coleta + rng.geometric(p_embarque) - 1
    transito = np.maximum(1, np.round(0.4 * leadtime + rng.normal(0, 1.6, size=total))).astype(int)
    dias_entrega = dias_embarque + transito

    data_coleta = _somar_uteis(solicitacao, dias_coleta, feriados)
    data_embarque = _somar_uteis(solicitacao, dias_embarque, feriados)
    data_entrega = _somar_uteis(solicitacao, dias_entrega, feriados)

    # 4. O que ainda não aconteceu na data de corte fica em branco; algumas OMs travam no caminho
    sem_coleta = (data_coleta > corte) | (rng.random(total) < 0.008)
    sem_embarque = sem_coleta | (data_embarque > corte)
    sem_entrega = sem_embarque | (data_entrega > corte) | (rng.random(total) < 0.008)

    # 5. Ocorrências
    ocorrencia = np.full(total, "", dtype=object)
    sorteio = rng.random(total)
    inicio_faixa = 0.0
    for nome, probabilidade in _OCORRENCIAS_SORTEADAS:
        faixa = (sorteio >= inicio_faixa) & (sorteio < inicio_faixa + probabilidade)
        ocorrencia[faixa] = nome
        inicio_faixa += probabilidade
    atraso_do_polo = coleta_atrasada & (ocorrencia == "") & (rng.random(total) < 0.12)
    ocorrencia[atraso_do_polo] = "ATRASO DEVIDO AO POLO"
    cancelada = np.isin(ocorrencia, ["OM CANCELADA", "COLETA CANCELADA DEVIDO A INVENTÁRIO"])
    sem_coleta, sem_embarque, sem_entrega = sem_coleta | cancelada, sem_embarque | cancelada, sem_entrega | cancelada

    # 6. Monta a tabela com as colunas do CONTROLE
    def como_datas(valores: np.ndarray, vazio: np.ndarray) -> pd.Series:
        serie = pd.Series(valores.astype("datetime64[ns]"))
        return serie.mask(vazio)

    oms["OM"] = cfg["numeracao"]["primeira_om"] + np.arange(total)
    oms["DATA DISP COLETA"] = pd.Series(disp.astype("datetime64[ns]"))
    oms["DATA SOLICITAÇÃO"] = pd.Series(solicitacao.astype("datetime64[ns]"))
    oms["DATA EFETIVA DA COLETA"] = como_datas(data_coleta, sem_coleta)
    oms["DATA DO EMBARQUE"] = como_datas(data_embarque, sem_embarque)
    oms["DATA EFETIVA ENTREGA"] = como_datas(data_entrega, sem_entrega)
    oms["OCORRÊNCIA"] = ocorrencia
    oms["DATA OCORRÊNCIA"] = oms["DATA EFETIVA DA COLETA"].fillna(oms["DATA SOLICITAÇÃO"]).mask(ocorrencia == "")

    coletada, embarcada = ~sem_coleta, ~sem_embarque
    oms["DACTE"] = pd.Series(rng.permutation(np.arange(100000, 100000 + total))).astype("string").where(coletada, "")
    oms["OST"] = ("OST" + pd.Series(rng.integers(10000, 99999, size=total)).astype("string")).where(coletada, "")
    oms["MINUTA DE EMBARQUE"] = ("MIN" + pd.Series(rng.integers(100000, 999999, size=total)).astype("string")).where(embarcada, "")
    observacoes = np.full(total, "", dtype=object)
    com_observacao = rng.random(total) < 0.05
    observacoes[com_observacao] = rng.choice(_OBSERVACOES, size=int(com_observacao.sum()))
    oms["OBSERVAÇÕES"] = observacoes

    return oms, perfis


def resumo_de_desempenho(oms: pd.DataFrame) -> pd.DataFrame:
    """Percentual de coletas e entregas no prazo por polo, com a mesma regra dos status do Excel."""
    feriados = _array_feriados(tabela_feriados()["DATA"])
    sol = oms["DATA SOLICITAÇÃO"].to_numpy().astype("datetime64[D]")
    prev_coleta = _somar_uteis(sol, oms["PRAZO_COLETA"].to_numpy(), feriados)
    prev_entrega = _somar_uteis(sol, oms["LEADTIME"].to_numpy(), feriados)
    sem_status_imposto = ~oms["OCORRÊNCIA"].isin([
        "COLETADO - TRANSP. ALTERNATIVA", "COLETA CANCELADA DEVIDO A INVENTÁRIO",
        "ATRASO DEVIDO AO POLO", "OM CANCELADA"])
    coleta = oms["DATA EFETIVA DA COLETA"].to_numpy().astype("datetime64[D]")
    entrega = oms["DATA EFETIVA ENTREGA"].to_numpy().astype("datetime64[D]")
    coletada = oms["DATA EFETIVA DA COLETA"].notna() & sem_status_imposto
    entregue = oms["DATA EFETIVA ENTREGA"].notna() & sem_status_imposto
    oms = oms.assign(
        COLETA_NO_PRAZO=pd.Series(coleta <= prev_coleta).where(coletada),
        ENTREGA_NO_PRAZO=pd.Series(entrega <= prev_entrega).where(entregue),
    )
    por_polo = oms.groupby("SIGLA").agg(
        OMs=("OM", "count"),
        pct_coleta_no_prazo=("COLETA_NO_PRAZO", lambda s: s.dropna().astype(float).mean()),
        pct_entrega_no_prazo=("ENTREGA_NO_PRAZO", lambda s: s.dropna().astype(float).mean()),
    )
    total = pd.DataFrame({
        "OMs": [len(oms)],
        "pct_coleta_no_prazo": [oms["COLETA_NO_PRAZO"].dropna().astype(float).mean()],
        "pct_entrega_no_prazo": [oms["ENTREGA_NO_PRAZO"].dropna().astype(float).mean()],
    }, index=["TOTAL"])
    return pd.concat([por_polo.sort_values("OMs", ascending=False), total])


def sinal_de_embarque(oms: pd.DataFrame, margem: int) -> dict[str, float]:
    """Mede se o sinal 'EMBARQUE EXCEDE O PRAZO' antecipa as entregas fora do prazo.

    O sinal vale quando (previsão de entrega - data do embarque) é menor ou igual à margem, em dias.
    """
    feriados = _array_feriados(tabela_feriados()["DATA"])
    sol = oms["DATA SOLICITAÇÃO"].to_numpy().astype("datetime64[D]")
    prev_entrega = pd.Series(_somar_uteis(sol, oms["LEADTIME"].to_numpy(), feriados))
    embarque = oms["DATA DO EMBARQUE"]
    entrega = oms["DATA EFETIVA ENTREGA"]
    base = embarque.notna() & entrega.notna() & ~oms["OCORRÊNCIA"].isin([
        "COLETADO - TRANSP. ALTERNATIVA", "COLETA CANCELADA DEVIDO A INVENTÁRIO",
        "ATRASO DEVIDO AO POLO", "OM CANCELADA"])
    folga = (prev_entrega - embarque).dt.days
    sinal = (folga <= margem) & base
    atrasada = (entrega > prev_entrega) & base
    return {
        "entregas_fora_do_prazo": int(atrasada.sum()),
        "parcela_dos_atrasos_com_sinal": float((sinal & atrasada).sum() / max(atrasada.sum(), 1)),
        "parcela_dos_sinais_que_atrasaram": float((sinal & atrasada).sum() / max(sinal.sum(), 1)),
    }