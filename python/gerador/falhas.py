"""Injeta falhas de qualidade conhecidas e mede quantas linhas cada verificação da aba QUALIDADE acusa."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendario import tabela_feriados
from .oms import COLUNAS_CONTROLE, _array_feriados, _somar_uteis

COLUNAS_FINAIS = COLUNAS_CONTROLE + ["SIGLA_MANUAL", "PREV_COLETA_MANUAL"]
_COLUNAS_DATA = [
    "DATA DISP COLETA", "DATA SOLICITAÇÃO", "DATA EFETIVA DA COLETA",
    "DATA DO EMBARQUE", "DATA EFETIVA ENTREGA", "DATA OCORRÊNCIA", "PREV_COLETA_MANUAL",
]
_EMBARQUES_ABSURDOS = ["2384-12-07", "2388-10-26", "2301-03-15", "13/02/206", "05/05/026"]

DESCRICOES = {
    "Q03": "Data de embarque absurda (ano distante ou texto)",
    "Q05": "Coleta efetiva anterior à solicitação",
    "Q06": "Embarque anterior à coleta efetiva",
    "Q07": "Entrega anterior ao embarque (inclui as da Q08)",
    "Q08": "Entrega anterior à solicitação",
    "Q09": "Entrega registrada sem data de embarque",
    "Q10": "Disponibilização posterior à solicitação",
    "Q11": "UF com espaço sobrando",
    "Q13": "OMs de cidades sem prazo no LEADTIME",
    "Q15": "QTD vazia",
    "Q16": "QTD igual a zero",
    "Q17": "Entrega registrada com DACTE vazio",
    "Q18": "Origens fora da base de municípios (documentadas como exceção)",
    "Exc_sigla": "Exceções manuais de polo (SIGLA_MANUAL)",
    "Exc_previsao": "Exceções manuais de previsão de coleta (PREV_COLETA_MANUAL)",
}


def _quantidade(rng: np.random.Generator, regra: dict, bases: dict[str, int]) -> int:
    """Sorteia quantas linhas receberão a falha: por faixa fixa ou por fração de uma base."""
    if "minimo" in regra:
        return int(rng.integers(regra["minimo"], regra["maximo"] + 1))
    fracao = rng.uniform(regra["frac_min"], regra["frac_max"])
    return int(round(fracao * bases[regra["base"]]))


def _grafia_diferente(nome: str) -> str:
    """Escreve o nome de outro jeito (abreviação ou troca de letra), como acontece nos relatórios."""
    for de, para in [("SANTA ", "STA. "), ("SANTO ", "STO. "), ("SAO ", "S. "), (" DO ", " D "), (" DA ", " D "), ("I", "Y")]:
        if de in nome:
            return nome.replace(de, para, 1)
    return nome[:-1]


def _renomear_origem(tabelas: dict, oms: pd.DataFrame, uf: str, antigo: str, novo: str) -> None:
    """Troca o nome de uma origem em todas as tabelas, mantendo tudo coerente."""
    oms.loc[(oms["ORIGEM"] == antigo) & (oms["UF"] == uf), "ORIGEM"] = novo
    for tabela, coluna in [("polos", "ORIGEM"), ("leadtime", "MUNICIPIO"), ("prazos_coleta_especial", "CIDADE")]:
        t = tabelas[tabela]
        t.loc[(t[coluna] == antigo) & (t["UF"] == uf), coluna] = novo


def _para_texto(serie: pd.Series) -> pd.Series:
    """Datas como texto 'aaaa-mm-dd' (vazio quando não há data)."""
    return serie.dt.strftime("%Y-%m-%d").fillna("")


def injetar_falhas(cfg: dict, tabelas: dict[str, pd.DataFrame], oms: pd.DataFrame, municipios: pd.DataFrame):
    """Aplica as falhas configuradas. Devolve (controle, tabelas_alteradas, excecoes, resumo_do_que_foi_injetado)."""
    rng = np.random.default_rng(cfg["semente"] + 2)
    regras = cfg["falhas"]
    oms = oms.copy()
    tabelas = {nome: t.copy() for nome, t in tabelas.items()}
    feriados = _array_feriados(tabelas["feriados"]["DATA"])
    injetado: dict[str, int] = {}

    contagem_cidade = oms.groupby(["ORIGEM", "UF"]).size()
    especiais = set(zip(tabelas["prazos_coleta_especial"]["CIDADE"], tabelas["prazos_coleta_especial"]["UF"]))
    cidades_usadas: set = set(especiais)

    # --- Q18: origens que não existem na lista de municípios (e são documentadas como exceção)
    total_origens = int(rng.integers(regras["Q18_origens_fora_da_base"]["minimo"], regras["Q18_origens_fora_da_base"]["maximo"] + 1))
    tipos = ["grafia", "distrito", "renda_extra"]
    tipos += [str(rng.choice(tipos)) for _ in range(total_origens - 3)]
    candidatas = [c for c, n in contagem_cidade.items() if 2 <= n <= 40 and c not in cidades_usadas]
    ordem = rng.permutation(len(candidatas))
    base_municipios = set(zip(municipios["CIDADE"], municipios["UF"]))
    excecoes = []
    for tipo in tipos:
        for posicao in ordem:
            cidade, uf = candidatas[posicao]
            if (cidade, uf) in cidades_usadas:
                continue
            if tipo == "grafia":
                novo = _grafia_diferente(cidade)
                motivo = "Grafia diferente da base oficial de municípios"
            elif tipo == "distrito":
                novo, motivo = f"DISTRITO DE {cidade}", "Distrito ou região administrativa: não consta como município na base"
            else:
                novo, motivo = f"PONTO RENDA EXTRA {len(excecoes) + 1:02d}", "Origem de renda extra (ponto de coleta sem município)"
            if (novo, uf) in base_municipios:
                continue
            _renomear_origem(tabelas, oms, uf, cidade, novo)
            cidades_usadas.add((cidade, uf))
            excecoes.append({"CHAVE": f"{novo}|{uf}", "MOTIVO": motivo})
            break
    injetado["Q18"] = len(excecoes)

    # --- Q13: cidades sem prazo cadastrado (saem da tabela LEADTIME)
    meta = int(rng.integers(regras["Q13_cidade_sem_prazo"]["minimo"], regras["Q13_cidade_sem_prazo"]["maximo"] + 1))
    contagem_cidade = oms.groupby(["ORIGEM", "UF"]).size()
    pequenas = [(c, n) for c, n in contagem_cidade.items() if n <= meta and c not in cidades_usadas]
    rng.shuffle(pequenas)
    sem_prazo, soma = [], 0
    for cidade, n in pequenas:
        if soma + n <= meta:
            sem_prazo.append(cidade)
            soma += n
        if soma == meta:
            break
    lt = tabelas["leadtime"]
    for cidade, uf in sem_prazo:
        lt = lt[~((lt["MUNICIPIO"] == cidade) & (lt["UF"] == uf))]
    tabelas["leadtime"] = lt.reset_index(drop=True)
    injetado["Q13"] = soma

    # --- Falhas de data (cada uma em linhas diferentes)
    total = len(oms)
    coletas, entregas = int(oms["DATA EFETIVA DA COLETA"].notna().sum()), int(oms["DATA EFETIVA ENTREGA"].notna().sum())
    bases = {"oms": total, "coletas": coletas, "entregas": entregas}
    usadas: set[int] = set()

    def sortear(condicao: pd.Series, quantidade: int) -> np.ndarray:
        disponiveis = np.array([i for i in oms.index[condicao] if i not in usadas])
        escolhidas = rng.choice(disponiveis, size=min(quantidade, len(disponiveis)), replace=False)
        usadas.update(int(i) for i in escolhidas)
        return escolhidas

    dia = np.timedelta64(1, "D")
    sol, coleta, embarque, entrega = (oms["DATA SOLICITAÇÃO"], oms["DATA EFETIVA DA COLETA"], oms["DATA DO EMBARQUE"], oms["DATA EFETIVA ENTREGA"])

    n_q08 = _quantidade(rng, regras["Q08_entrega_antes_solicitacao"], bases)
    n_q07 = max(_quantidade(rng, regras["Q07_entrega_antes_embarque"], bases), n_q08 + 1)  # as linhas da Q08 também entram na Q07
    n_q06 = _quantidade(rng, regras["Q06_embarque_antes_coleta"], bases)
    n_q05 = _quantidade(rng, regras["Q05_coleta_antes_solicitacao"], bases)
    n_q09 = _quantidade(rng, regras["Q09_entrega_sem_embarque"], bases)
    n_q10 = _quantidade(rng, regras["Q10_disp_depois_solicitacao"], bases)
    n_q03 = _quantidade(rng, regras["Q03_embarque_absurdo"], bases)

    for i in sortear(entrega.notna() & embarque.notna(), n_q08):
        oms.loc[i, "DATA EFETIVA ENTREGA"] = sol[i] - int(rng.integers(1, 4)) * dia
    for i in sortear(entrega.notna() & embarque.notna() & (embarque >= sol + 4 * dia), n_q07 - n_q08):
        oms.loc[i, "DATA EFETIVA ENTREGA"] = embarque[i] - int(rng.integers(1, 4)) * dia
    for i in sortear(embarque.notna() & coleta.notna(), n_q06):
        oms.loc[i, "DATA DO EMBARQUE"] = coleta[i] - int(rng.integers(1, 4)) * dia
    for i in sortear(coleta.notna(), n_q05):
        oms.loc[i, "DATA EFETIVA DA COLETA"] = sol[i] - int(rng.integers(1, 5)) * dia
    for i in sortear(entrega.notna() & embarque.notna(), n_q09):
        oms.loc[i, "DATA DO EMBARQUE"] = pd.NaT
    for i in sortear(oms["DATA SOLICITAÇÃO"].notna(), n_q10):
        oms.loc[i, "DATA DISP COLETA"] = sol[i] + int(rng.integers(1, 6)) * dia
    injetado.update({"Q05": n_q05, "Q06": n_q06, "Q07": n_q07, "Q08": n_q08, "Q09": n_q09, "Q10": n_q10})

    # --- Quantidade, DACTE e espaços na UF
    n_q16 = _quantidade(rng, regras["Q16_qtd_zero"], bases)
    n_q15 = _quantidade(rng, regras["Q15_qtd_vazia"], bases)
    pool = rng.permutation(oms.index.to_numpy())
    oms["QTD"] = oms["QTD"].astype(float)
    oms.loc[pool[:n_q16], "QTD"] = 0
    oms.loc[pool[n_q16:n_q16 + n_q15], "QTD"] = np.nan
    comentregas = oms.index[oms["DATA EFETIVA ENTREGA"].notna()]
    n_q17 = _quantidade(rng, regras["Q17_dacte_vazio"], bases)
    oms.loc[rng.choice(comentregas, size=n_q17, replace=False), "DACTE"] = ""
    n_q11 = _quantidade(rng, regras["Q11_espaco_em_uf"], bases)
    for i in rng.choice(oms.index.to_numpy(), size=n_q11, replace=False):
        oms.loc[i, "UF"] = (" " + oms.loc[i, "UF"]) if rng.random() < 0.4 else (oms.loc[i, "UF"] + " ")
    injetado.update({"Q15": n_q15, "Q16": n_q16, "Q17": n_q17, "Q11": n_q11})

    # --- Exceções manuais (polo e previsão de coleta digitados à mão, com o motivo registrado)
    oms["SIGLA_MANUAL"] = ""
    oms["PREV_COLETA_MANUAL"] = pd.NaT
    n_sigla = _quantidade(rng, regras["excecoes_manuais"]["sigla"], bases)
    n_prev = _quantidade(rng, regras["excecoes_manuais"]["previsao_coleta"], bases)
    polos_existentes = sorted(tabelas["polos"]["SIGLA"].unique())
    for i in rng.choice(oms.index.to_numpy(), size=n_sigla, replace=False):
        outros = [p for p in polos_existentes if p != oms.loc[i, "SIGLA"]]
        oms.loc[i, "SIGLA_MANUAL"] = str(rng.choice(outros))
        oms.loc[i, "OBSERVAÇÕES"] = "Exceção: polo definido manualmente"
    for i in rng.choice(oms.index[coleta.notna()].to_numpy(), size=n_prev, replace=False):
        previsto = _somar_uteis(np.array([sol[i].to_datetime64().astype("datetime64[D]")]), np.array([int(oms.loc[i, "PRAZO_COLETA"]) + 2]), feriados)[0]
        oms.loc[i, "PREV_COLETA_MANUAL"] = pd.Timestamp(previsto)
        oms.loc[i, "OBSERVAÇÕES"] = "Exceção: previsão de coleta combinada com o polo"
    injetado.update({"Exc_sigla": n_sigla, "Exc_previsao": n_prev})

    # --- Texto final: datas viram texto; Q03 troca algumas datas de embarque por valores absurdos
    controle = oms.copy()
    for coluna in _COLUNAS_DATA:
        controle[coluna] = _para_texto(pd.to_datetime(controle[coluna]))
    candidatos_q03 = [i for i in controle.index[controle["DATA DO EMBARQUE"] != ""] if i not in usadas]
    for i in rng.choice(candidatos_q03, size=n_q03, replace=False):
        controle.loc[i, "DATA DO EMBARQUE"] = str(rng.choice(_EMBARQUES_ABSURDOS))
    injetado["Q03"] = n_q03

    controle["QTD"] = controle["QTD"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    controle = controle[COLUNAS_FINAIS + ["SIGLA", "LEADTIME", "PRAZO_COLETA"]]
    resumo = pd.DataFrame({"CODIGO": list(injetado), "QUANTIDADE_INJETADA": list(injetado.values())})
    return controle, tabelas, pd.DataFrame(excecoes), resumo


def _data_valida(serie: pd.Series, ano_min: int, ano_max: int) -> tuple[pd.Series, pd.Series]:
    """Devolve (datas, preenchida_mas_invalida). Texto fora do formato ou ano fora da janela é inválido."""
    texto = serie.astype("string").fillna("")
    formato_ok = texto.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    ano = pd.to_numeric(texto.str[:4], errors="coerce")
    na_janela = formato_ok & ano.between(ano_min, ano_max)
    datas = pd.to_datetime(texto.where(na_janela), format="%Y-%m-%d", errors="coerce")
    preenchida = texto != ""
    return datas, preenchida & ~na_janela


def medir_falhas(controle: pd.DataFrame, tabelas: dict[str, pd.DataFrame], municipios: pd.DataFrame, excecoes: pd.DataFrame) -> dict[str, int]:
    """Conta as linhas que cada verificação da aba QUALIDADE acusaria (mesmas regras do Excel)."""
    sol, _ = _data_valida(controle["DATA SOLICITAÇÃO"], 2025, 2027)
    disp, _ = _data_valida(controle["DATA DISP COLETA"], 2025, 2027)
    coleta, _ = _data_valida(controle["DATA EFETIVA DA COLETA"], 2026, 2027)
    embarque, embarque_invalido = _data_valida(controle["DATA DO EMBARQUE"], 2026, 2027)
    entrega, _ = _data_valida(controle["DATA EFETIVA ENTREGA"], 2026, 2027)
    tem_entrega = controle["DATA EFETIVA ENTREGA"] != ""

    medidas = {
        "Q03": int(embarque_invalido.sum()),
        "Q05": int((coleta < sol).sum()),
        "Q06": int((embarque < coleta).sum()),
        "Q07": int((entrega < embarque).sum()),
        "Q08": int((entrega < sol).sum()),
        "Q09": int((tem_entrega & (controle["DATA DO EMBARQUE"] == "")).sum()),
        "Q10": int((disp > sol).sum()),
        "Q11": int((controle["UF"] != controle["UF"].str.strip()).sum()),
        "Q15": int((controle["QTD"] == "").sum()),
        "Q16": int((controle["QTD"] == "0").sum()),
        "Q17": int((tem_entrega & (controle["DACTE"] == "")).sum()),
    }
    chaves_prazo = set(zip(tabelas["leadtime"]["MUNICIPIO"], tabelas["leadtime"]["UF"]))
    chave_oms = list(zip(controle["ORIGEM"], controle["UF"].str.strip()))
    medidas["Q13"] = int(sum(1 for c in chave_oms if c not in chaves_prazo))
    base = set(zip(municipios["CIDADE"], municipios["UF"]))
    fora = {c for c in chave_oms if c not in base}
    medidas["Q18_origens"] = len(fora)
    documentadas = set(excecoes["CHAVE"]) if len(excecoes) else set()
    medidas["Q18"] = int(sum(1 for (cidade, uf) in chave_oms if (cidade, uf) not in base and f"{cidade}|{uf}" not in documentadas))
    return medidas


def montar_resumo(injetado: pd.DataFrame, medidas: dict[str, int], controle: pd.DataFrame) -> pd.DataFrame:
    """Tabela com o que foi injetado e o que as verificações medem, para o README e para os testes."""
    injetado = injetado.set_index("CODIGO")["QUANTIDADE_INJETADA"]
    medido = dict(medidas)
    medido["Q18"] = medidas["Q18_origens"]
    medido["Exc_sigla"] = int((controle["SIGLA_MANUAL"] != "").sum())
    medido["Exc_previsao"] = int((controle["PREV_COLETA_MANUAL"] != "").sum())
    linhas = [
        {"CODIGO": codigo, "DESCRICAO": texto, "INJETADO": int(injetado[codigo]), "MEDIDO": int(medido[codigo])}
        for codigo, texto in DESCRICOES.items()
    ]
    linhas.append({"CODIGO": "Q18 após exceções", "DESCRICAO": "OMs de origens fora da base e sem exceção documentada", "INJETADO": 0, "MEDIDO": int(medidas["Q18"])})
    return pd.DataFrame(linhas)