"""Ponto de entrada do gerador de dados sintéticos."""
from gerador.config import RAIZ, carregar_config
from gerador.dimensoes import carregar_municipios, gerar_dimensoes


def main() -> None:
    cfg = carregar_config()
    municipios = carregar_municipios(RAIZ / "data" / "reference" / "municipios.xlsx")
    tabelas = gerar_dimensoes(cfg, municipios)

    destino = RAIZ / "data" / "synthetic"
    destino.mkdir(parents=True, exist_ok=True)
    for nome, tabela in tabelas.items():
        tabela.to_csv(destino / f"{nome}.csv", index=False, encoding="utf-8-sig")
        print(f"{nome:<24} {len(tabela):>5} linhas -> data/synthetic/{nome}.csv")


if __name__ == "__main__":
    main()