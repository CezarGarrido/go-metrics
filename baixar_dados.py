#!/usr/bin/env python3
"""
Script para baixar dados reais de preços de combustíveis da ANP
Dataset: Série Histórica de Preços de Combustíveis e de GLP
Fonte: https://dados.gov.br/dados/conjuntos-dados/serie-historica-de-precos-de-combustiveis-e-de-glp
"""

import os
from pathlib import Path

import requests

# Criar diretórios
Path("data/raw").mkdir(parents=True, exist_ok=True)

# URLs dos arquivos CSV da ANP (2024)
# Fonte: Portal de Dados Abertos da ANP
# Padrão: https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/arquivos/shpc/dsan/2024/precos-gasolina-etanol-{MES}.csv
arquivos = []
for mes in range(1, 13):
    mes_str = f"{mes:02d}"
    nome_arquivo = f"2024_gasolina_etanol_{mes_str}.csv"
    url = f"https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/arquivos/shpc/dsan/2024/precos-gasolina-etanol-{mes_str}.csv"
    arquivos.append((nome_arquivo, url))

print("=" * 60)
print("DOWNLOAD DE DADOS DA ANP - PREÇOS DE COMBUSTÍVEIS 2024")
print("=" * 60)
print(f"Total de arquivos: {len(arquivos)}")
print()

sucesso = 0
erros = 0

for i, (nome_arquivo, url) in enumerate(arquivos, 1):
    caminho_destino = f"data/raw/{nome_arquivo}"

    print(f"[{i:2d}/{len(arquivos)}] Baixando {nome_arquivo}...", end=" ")

    try:
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            with open(caminho_destino, "wb") as f:
                f.write(response.content)

            tamanho_mb = len(response.content) / (1024 * 1024)
            print(f"✓ Sucesso! ({tamanho_mb:.2f} MB)")
            sucesso += 1
        else:
            print(f"✗ Erro HTTP {response.status_code}")
            erros += 1

    except Exception as e:
        print(f"✗ Erro: {str(e)}")
        erros += 1

print()
print("=" * 60)
print("RESUMO DO DOWNLOAD")
print("=" * 60)
print(f"✓ Sucesso: {sucesso} arquivo(s)")
print(f"✗ Erros:   {erros} arquivo(s)")
print(f"📂 Destino: {os.path.abspath('data/raw/')}")
print()

if sucesso > 0:
    print("=" * 60)
    print("PRÓXIMOS PASSOS")
    print("=" * 60)
    print("1. Execute: python 01_pipeline_duckdb.py")
    print("2. Execute: python 02_consultas_sql_analiticas.py")
    print("3. Execute: python 03_modelagem_preditiva_fixed.py")
    print("=" * 60)
else:
    print("⚠️  ATENÇÃO: Nenhum arquivo foi baixado com sucesso!")
    print("   Verifique sua conexão com a internet e tente novamente.")
