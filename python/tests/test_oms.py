import numpy as np
import pandas as pd
import pytest

from gerador.config import carregar_config
from gerador.dimensoes import gerar_dimensoes
from gerador.oms import COLUNAS_CONTROLE, gerar_oms, resumo_de_desempenho, sinal_de_embarque


@pytest.fixture(scope="module")
def cfg() -> dict:
    return carregar_config()


@pytest.fixture(scope="module")
def municipios_falsos() -> pd.DataFrame:
    linhas = [(f"U{u:02d}", f"CIDADE {u:02d}-{c:02d}") for u in range(1, 16) for c in range(1, 31)]
    return pd.DataFrame(linhas, columns=["UF", "CIDADE"])


@pytest.fixture(scope="module")
def resultado(cfg, municipios_falsos):
    tabelas = gerar_dimensoes(cfg, municipios_falsos)
    oms, perfis = gerar_oms(cfg, tabelas)
    return oms, perfis, tabelas


def test_quantidade_e_numeracao(resultado, cfg):
    oms = resultado[0]
    assert len(oms) == cfg["volume"]["total_oms"]
    assert oms["OM"].is_unique
    assert oms["OM"].min() == cfg["numeracao"]["primeira_om"]
    assert oms["OM"].is_monotonic_increasing


def test_tem_as_colunas_do_controle(resultado):
    assert set(COLUNAS_CONTROLE) <= set(resultado[0].columns)


def test_solicitacoes_sao_dias_uteis_dentro_do_periodo(resultado, cfg):
    solicitacao = resultado[0]["DATA SOLICITAÇÃO"]
    assert solicitacao.min() >= pd.Timestamp(cfg["periodo"]["inicio"])
    assert solicitacao.max() <= pd.Timestamp(cfg["periodo"]["fim"])
    assert np.is_busday(solicitacao.to_numpy().astype("datetime64[D]")).all()


def test_sequencia_das_datas(resultado):
    oms = resultado[0]
    coleta, embarque, entrega = oms["DATA EFETIVA DA COLETA"], oms["DATA DO EMBARQUE"], oms["DATA EFETIVA ENTREGA"]
    assert (coleta.dropna() >= oms.loc[coleta.notna(), "DATA SOLICITAÇÃO"]).all()
    assert (embarque.dropna() >= coleta[embarque.notna()]).all()
    assert (entrega.dropna() > embarque[entrega.notna()]).all()
    assert not (embarque.notna() & coleta.isna()).any()
    assert not (entrega.notna() & embarque.isna()).any()


def test_nada_acontece_depois_da_data_de_corte(resultado, cfg):
    oms, corte = resultado[0], pd.Timestamp(cfg["periodo"]["fim"])
    for coluna in ["DATA EFETIVA DA COLETA", "DATA DO EMBARQUE", "DATA EFETIVA ENTREGA"]:
        assert (oms[coluna].dropna() <= corte).all()


def test_quantidade_de_pos_e_disponibilizacao(resultado):
    oms = resultado[0]
    assert oms["QTD"].min() >= 1
    assert (oms["DATA DISP COLETA"] <= oms["DATA SOLICITAÇÃO"]).all()


def test_dacte_nao_se_repete(resultado):
    dacte = resultado[0]["DACTE"]
    assert dacte[dacte != ""].is_unique


def test_desempenho_dentro_de_faixas_plausiveis(resultado):
    resumo = resumo_de_desempenho(resultado[0])
    total = resumo.loc["TOTAL"]
    assert 0.75 <= total["pct_coleta_no_prazo"] <= 0.95
    assert 0.50 <= total["pct_entrega_no_prazo"] <= 0.75


def test_existe_um_polo_que_coleta_bem_e_entrega_mal(resultado):
    oms, perfis, _ = resultado
    resumo = resumo_de_desempenho(oms)
    polo = perfis.index[perfis["PERFIL"] == "problema"][0]
    assert resumo.loc[polo, "pct_coleta_no_prazo"] >= 0.80
    assert resumo.loc[polo, "pct_entrega_no_prazo"] <= 0.40


def test_sinal_de_embarque_antecipa_a_maior_parte_dos_atrasos(resultado, cfg):
    sinal = sinal_de_embarque(resultado[0], cfg["prazos"]["margem_embarque_dias"])
    assert sinal["parcela_dos_atrasos_com_sinal"] >= 0.60


def test_mesma_semente_gera_as_mesmas_oms(cfg, municipios_falsos):
    tabelas = gerar_dimensoes(cfg, municipios_falsos)
    primeira, _ = gerar_oms(cfg, tabelas)
    segunda, _ = gerar_oms(cfg, tabelas)
    pd.testing.assert_frame_equal(primeira, segunda)