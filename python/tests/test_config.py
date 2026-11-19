from gerador.config import carregar_config


def test_config_tem_os_parametros_aprovados():
    cfg = carregar_config()
    assert cfg["semente"] == 42
    assert cfg["periodo"]["inicio"] == "2026-01-05"
    assert cfg["periodo"]["fim"] == "2026-09-17"
    assert 1800 <= cfg["volume"]["total_oms"] <= 2200
    assert 12 <= cfg["estrutura"]["quantidade_polos"] <= 15
    assert 10 <= cfg["estrutura"]["quantidade_ufs"] <= 13
    assert cfg["numeracao"]["primeira_om"] == 9000001