import math

import pandas as pd
import pytest

from gerador.config import carregar_config
from gerador.dimensoes import gerar_dimensoes
from gerador.falhas import COLUNAS_FINAIS, injetar_falhas, medir_falhas, montar_resumo
from gerador.oms import gerar_oms


@pytest.fixture(scope="module")
def cfg() -> dict:
    return carregar_config()


@pytest.fixture(scope="module")
def municipios_falsos() -> pd.DataFrame:
    linhas = [(f"U{u:02d}", f"CIDADE {u:02d}-{c:02d}") for u in range(1, 16) for c in range(1, 31)]
    return pd.DataFrame(linhas, columns=["UF", "CIDADE"])


@pytest.fixture(scope="module")
def cenario(cfg, municipios_falsos):
    tabelas = gerar_dimensoes(cfg, municipios_falsos)
    oms, _ = gerar_oms(cfg, tabelas)
    controle, tabelas_novas, excecoes, injetado = injetar_falhas(cfg, tabelas, oms, municipios_falsos)
    medidas = medir_falhas(controle, tabelas_novas, municipios_falsos, excecoes)
    return {"controle": controle, "tabelas": tabelas_novas, "excecoes": excecoes,
            "injetado": injetado, "medidas": medidas, "oms": oms}


def test_colunas_e_quantidade_de_linhas(cenario, cfg):
    controle = cenario["controle"]
    assert len(controle) == cfg["volume"]["total_oms"]
    assert set(COLUNAS_FINAIS) <= set(controle.columns)
    assert controle["OM"].is_unique


@pytest.mark.parametrize("codigo", ["Q03", "Q06", "Q07", "Q08", "Q10", "Q11", "Q13", "Q15", "Q17"])
def test_falhas_por_faixa_fixa_dentro_do_limite(cenario, cfg, codigo):
    regra = next(r for chave, r in cfg["falhas"].items() if chave.startswith(codigo + "_"))
    assert regra["minimo"] <= cenario["medidas"][codigo] <= regra["maximo"]


def test_falhas_por_fracao_dentro_do_limite(cenario, cfg):
    controle, medidas = cenario["controle"], cenario["medidas"]
    total = len(controle)
    coletas = int((cenario["oms"]["DATA EFETIVA DA COLETA"].notna()).sum())
    entregas = int((cenario["oms"]["DATA EFETIVA ENTREGA"].notna()).sum())
    q05, q09, q16 = cfg["falhas"]["Q05_coleta_antes_solicitacao"], cfg["falhas"]["Q09_entrega_sem_embarque"], cfg["falhas"]["Q16_qtd_zero"]
    assert math.floor(q05["frac_min"] * coletas) <= medidas["Q05"] <= math.ceil(q05["frac_max"] * coletas)
    assert math.floor(q09["frac_min"] * entregas) <= medidas["Q09"] <= math.ceil(q09["frac_max"] * entregas)
    assert math.floor(q16["frac_min"] * total) <= medidas["Q16"] <= math.ceil(q16["frac_max"] * total)


def test_o_que_foi_medido_bate_com_o_que_foi_injetado(cenario):
    resumo = montar_resumo(cenario["injetado"], cenario["medidas"], cenario["controle"])
    base = resumo[~resumo["CODIGO"].isin(["Q18 após exceções"])]
    assert (base["INJETADO"] == base["MEDIDO"]).all()


def test_origens_fora_da_base_estao_documentadas(cenario, cfg):
    regra = cfg["falhas"]["Q18_origens_fora_da_base"]
    assert regra["minimo"] <= cenario["medidas"]["Q18_origens"] <= regra["maximo"]
    assert cenario["medidas"]["Q18"] == 0  # depois de documentadas, nenhuma OM fica sem exceção
    assert len(cenario["excecoes"]) == cenario["medidas"]["Q18_origens"]


def test_excecoes_manuais(cenario, cfg):
    controle = cenario["controle"]
    sigla, previsao = cfg["falhas"]["excecoes_manuais"]["sigla"], cfg["falhas"]["excecoes_manuais"]["previsao_coleta"]
    assert sigla["minimo"] <= (controle["SIGLA_MANUAL"] != "").sum() <= sigla["maximo"]
    assert previsao["minimo"] <= (controle["PREV_COLETA_MANUAL"] != "").sum() <= previsao["maximo"]
    com_excecao = controle[(controle["SIGLA_MANUAL"] != "") | (controle["PREV_COLETA_MANUAL"] != "")]
    assert com_excecao["OBSERVAÇÕES"].str.startswith("Exceção").all()  # o motivo está registrado


def test_datas_absurdas_existem_como_texto(cenario):
    embarque = cenario["controle"]["DATA DO EMBARQUE"]
    assert (embarque.str.len() != 10).sum() + embarque.str.startswith(("23", "24")).sum() > 0


def test_mesma_semente_gera_as_mesmas_falhas(cfg, municipios_falsos):
    tabelas = gerar_dimensoes(cfg, municipios_falsos)
    oms, _ = gerar_oms(cfg, tabelas)
    primeira = injetar_falhas(cfg, tabelas, oms, municipios_falsos)[0]
    segunda = injetar_falhas(cfg, tabelas, oms, municipios_falsos)[0]
    pd.testing.assert_frame_equal(primeira, segunda)