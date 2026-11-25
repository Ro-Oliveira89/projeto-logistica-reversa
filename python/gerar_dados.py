"""Ponto de entrada do gerador de dados sintéticos."""
from gerador.config import RAIZ, carregar_config
from gerador.dimensoes import carregar_municipios, gerar_dimensoes
from gerador.oms import COLUNAS_CONTROLE, gerar_oms, resumo_de_desempenho, sinal_de_embarque


def main() -> None:
    cfg = carregar_config()
    municipios = carregar_municipios(RAIZ / "data" / "reference" / "municipios.xlsx")
    destino = RAIZ / "data" / "synthetic"
    destino.mkdir(parents=True, exist_ok=True)

    # Tabelas de apoio (polos, LEADTIME, prazos especiais, feriados e ocorrências)
    tabelas = gerar_dimensoes(cfg, municipios)
    for nome, tabela in tabelas.items():
        tabela.to_csv(destino / f"{nome}.csv", index=False, encoding="utf-8-sig")
        print(f"{nome:<24} {len(tabela):>5} linhas -> data/synthetic/{nome}.csv")

    # OMs e o caminho de cada uma (coleta, embarque e entrega)
    oms, perfis = gerar_oms(cfg, tabelas)
    oms[COLUNAS_CONTROLE].to_csv(destino / "oms_base.csv", index=False, encoding="utf-8-sig")
    print(f"{'oms_base':<24} {len(oms):>5} linhas -> data/synthetic/oms_base.csv")

    # Conferência rápida do desempenho simulado
    resumo = resumo_de_desempenho(oms)
    total = resumo.loc["TOTAL"]
    print(f"\nColetas no prazo:  {total['pct_coleta_no_prazo']:.1%}")
    print(f"Entregas no prazo: {total['pct_entrega_no_prazo']:.1%}")
    polo_problema = perfis.index[perfis["PERFIL"] == "problema"][0]
    linha = resumo.loc[polo_problema]
    print(f"Polo problema: {polo_problema} (coleta {linha['pct_coleta_no_prazo']:.1%}, entrega {linha['pct_entrega_no_prazo']:.1%})")
    sinal = sinal_de_embarque(oms, cfg["prazos"]["margem_embarque_dias"])
    print(f"Atrasos de entrega com o sinal de embarque: {sinal['parcela_dos_atrasos_com_sinal']:.1%}")


if __name__ == "__main__":
    main()