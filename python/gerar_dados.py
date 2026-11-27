"""Ponto de entrada do gerador de dados sintéticos."""
from gerador.config import RAIZ, carregar_config
from gerador.dimensoes import carregar_municipios, gerar_dimensoes
from gerador.falhas import COLUNAS_FINAIS, injetar_falhas, medir_falhas, montar_resumo
from gerador.oms import COLUNAS_CONTROLE, gerar_oms, resumo_de_desempenho, sinal_de_embarque


def salvar(tabela, nome: str, destino) -> None:
    tabela.to_csv(destino / f"{nome}.csv", index=False, encoding="utf-8-sig")
    print(f"{nome:<24} {len(tabela):>5} linhas -> data/synthetic/{nome}.csv")


def main() -> None:
    cfg = carregar_config()
    municipios = carregar_municipios(RAIZ / "data" / "reference" / "municipios.xlsx")
    destino = RAIZ / "data" / "synthetic"
    destino.mkdir(parents=True, exist_ok=True)

    # 1. Tabelas de apoio e OMs com o caminho de cada uma (base limpa, sem falhas)
    tabelas = gerar_dimensoes(cfg, municipios)
    oms, perfis = gerar_oms(cfg, tabelas)
    oms[COLUNAS_CONTROLE].to_csv(destino / "oms_base.csv", index=False, encoding="utf-8-sig")

    # 2. Falhas injetadas (para a aba QUALIDADE do Excel público)
    controle, tabelas, excecoes, injetado = injetar_falhas(cfg, tabelas, oms, municipios)
    medidas = medir_falhas(controle, tabelas, municipios, excecoes)
    resumo = montar_resumo(injetado, medidas, controle)

    # 3. Arquivos finais
    for nome, tabela in tabelas.items():
        salvar(tabela, nome, destino)
    salvar(excecoes, "excecoes", destino)
    salvar(controle[COLUNAS_FINAIS], "controle_sintetico", destino)
    salvar(resumo, "falhas_injetadas", destino)

    # 4. Conferências
    print("\nFalhas (injetado x medido):")
    for linha in resumo.itertuples():
        marca = "" if linha.INJETADO == linha.MEDIDO or linha.CODIGO.startswith("Q07") else "  <-- diferente"
        print(f"  {linha.CODIGO:<18} injetado {linha.INJETADO:>4} | medido {linha.MEDIDO:>4}{marca}")
    desempenho = resumo_de_desempenho(oms).loc["TOTAL"]
    print(f"\nColetas no prazo (base limpa):  {desempenho['pct_coleta_no_prazo']:.1%}")
    print(f"Entregas no prazo (base limpa): {desempenho['pct_entrega_no_prazo']:.1%}")
    polo = perfis.index[perfis["PERFIL"] == "problema"][0]
    print(f"Polo problema: {polo}")
    print(f"Atrasos com o sinal de embarque: {sinal_de_embarque(oms, cfg['prazos']['margem_embarque_dias'])['parcela_dos_atrasos_com_sinal']:.1%}")


if __name__ == "__main__":
    main()