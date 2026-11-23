import pandas as pd
import pytest

from gerador.config import carregar_config
from gerador.dimensoes import gerar_dimensoes


@pytest.fixture
def municipios_falsos() -> pd.DataFrame:
    """Lista pequena e inventada de municípios, para o teste não depender do arquivo real."""
    linhas = [(f"U{u:02d}", f"CIDADE {u:02d}-{c:02d}") for u in range(1, 16) for c in range(1, 31)]
    return pd.DataFrame(linhas, columns=["UF", "CIDADE"])


@pytest.fixture
def tabelas(municipios_falsos):
    return gerar_dimensoes(carregar_config(), municipios_falsos)


def test_mesma_semente_gera_o_mesmo_resultado(municipios_falsos):
    cfg = carregar_config()
    primeira = gerar_dimensoes(cfg, municipios_falsos)
    segunda = gerar_dimensoes(cfg, municipios_falsos)
    for nome in primeira:
        pd.testing.assert_frame_equal(primeira[nome], segunda[nome])


def test_quantidades_seguem_a_configuracao(tabelas):
    cfg = carregar_config()
    polos = tabelas["polos"]
    assert len(polos) == cfg["estrutura"]["cidades_atendidas"]
    assert polos["UF"].nunique() == cfg["estrutura"]["quantidade_ufs"]
    assert polos["SIGLA"].nunique() == cfg["estrutura"]["quantidade_polos"]


def test_cidade_e_uf_nao_se_repetem(tabelas):
    assert not tabelas["polos"].duplicated(subset=["ORIGEM", "UF"]).any()


def test_nomes_sao_ficticios(tabelas):
    polos = tabelas["polos"]
    assert polos["SIGLA"].str.startswith("POLO_").all()
    assert polos["REPRESENTANTE"].str.startswith("REP_").all()
    assert polos["MODAL"].str.startswith("MODAL_").all()


def test_leadtime_dentro_dos_limites_e_ligado_aos_polos(tabelas):
    cfg = carregar_config()["prazos"]["leadtime"]
    leadtime = tabelas["leadtime"]
    assert leadtime["TRANSIT_DIAS_UTEIS"].between(cfg["dias_minimo"], cfg["dias_maximo"]).all()
    chaves_polos = set(zip(tabelas["polos"]["ORIGEM"], tabelas["polos"]["UF"]))
    assert set(zip(leadtime["MUNICIPIO"], leadtime["UF"])) == chaves_polos


def test_prazos_especiais(tabelas):
    cfg = carregar_config()["prazos"]["coleta_especial"]
    especiais = tabelas["prazos_coleta_especial"]
    assert len(especiais) == cfg["quantidade_cidades"]
    assert especiais["PRAZO"].between(cfg["dias_minimo"], cfg["dias_maximo"]).all()
    chaves_leadtime = set(zip(tabelas["leadtime"]["MUNICIPIO"], tabelas["leadtime"]["UF"]))
    assert set(zip(especiais["CIDADE"], especiais["UF"])) <= chaves_leadtime


def test_ocorrencias(tabelas):
    ocorrencias = tabelas["ocorrencias"]
    assert len(ocorrencias) == 21
    assert not ocorrencias["OCORRÊNCIA"].duplicated().any()
    assert (ocorrencias["STATUS COLETA"] != "").sum() == 4  # só quatro impõem status