#!/usr/bin/env python3
"""
Pipeline de Integração e Limpeza de Dados com DuckDB
Dataset: Série Histórica de Preços de Combustíveis - ANP 2024
Autor: Trabalho Final de Data Science
Data: Novembro 2025
"""

import glob

import duckdb

print("=" * 80)
print("PIPELINE DE INTEGRAÇÃO E LIMPEZA - DUCKDB")
print("Dataset: Preços de Combustíveis ANP 2024")
print("=" * 80)

# Criar conexão com DuckDB
conn = duckdb.connect("data/combustiveis.duckdb")

print("\n[1/6] Criando banco de dados DuckDB...")

# Criar tabela para dados brutos
conn.execute("""
    CREATE OR REPLACE TABLE precos_brutos (
        regiao_sigla VARCHAR,
        estado_sigla VARCHAR,
        municipio VARCHAR,
        revenda VARCHAR,
        cnpj_revenda VARCHAR,
        nome_rua VARCHAR,
        numero_rua VARCHAR,
        complemento VARCHAR,
        bairro VARCHAR,
        cep VARCHAR,
        produto VARCHAR,
        data_coleta VARCHAR,
        valor_venda VARCHAR,
        valor_compra DOUBLE,
        unidade_medida VARCHAR,
        bandeira VARCHAR,
        mes_ref INTEGER

    );
""")

print("✓ Tabela 'precos_brutos' criada com sucesso.")

# Inserir dados dos CSVs na tabela
print("\n[2/6] Inserindo dados dos CSVs na tabela...")

csv_files = glob.glob("data/raw/2024_gasolina_etanol_*.csv")

for i, file in enumerate(csv_files, 1):
    mes = int(file.split("_")[-1].split(".")[0])
    print(f"  > Processando arquivo {i}/{len(csv_files)}: {file} (Mês: {mes})")
    conn.execute(f"""
        INSERT INTO precos_brutos
        SELECT *, {mes} as mes_ref FROM read_csv_auto('{file}', sep=';', header=true, ignore_errors=true);
    """)

print("✓ Dados inseridos com sucesso.")

# Limpeza e transformação dos dados
print("\n[3/6] Limpando e transformando os dados...")

conn.execute("""
    CREATE OR REPLACE TABLE precos_tratados AS
    SELECT
        regiao_sigla,
        estado_sigla,
        municipio,
        revenda,
        cnpj_revenda,
        nome_rua,
        numero_rua,
        complemento,
        bairro,
        cep,
        produto,
        CASE WHEN TRY_STRPTIME(data_coleta, '%d/%m/%Y') IS NOT NULL THEN TRY_STRPTIME(data_coleta, '%d/%m/%Y') ELSE TRY_STRPTIME(data_coleta, '%Y-%m-%d') END as data_coleta,
        CAST(REPLACE(valor_venda, ',', '.') AS DOUBLE) as valor_venda,
        valor_compra,
        unidade_medida,
        bandeira,
        mes_ref
    FROM precos_brutos
    WHERE valor_venda IS NOT NULL;
""")

print("✓ Dados limpos e transformados com sucesso.")

# Análise exploratória inicial
print("\n[4/6] Análise exploratória inicial...")

print("  > Contagem de registros por produto:")
print(
    conn.execute(
        "SELECT produto, COUNT(*) as total FROM precos_tratados GROUP BY produto ORDER BY total DESC"
    ).fetchdf()
)

print("\n  > Média de preço de venda por produto:")
print(
    conn.execute(
        "SELECT produto, AVG(valor_venda) as media_preco FROM precos_tratados GROUP BY produto ORDER BY media_preco DESC"
    ).fetchdf()
)

print("\n  > Top 10 municípios com gasolina mais cara:")
print(
    conn.execute("""
    SELECT municipio, estado_sigla, AVG(valor_venda) as media_preco
    FROM precos_tratados
    WHERE produto = 'GASOLINA'
    GROUP BY municipio, estado_sigla
    ORDER BY media_preco DESC
    LIMIT 10;
""").fetchdf()
)

# Exportar dados limpos para Parquet
print("\n[5/6] Exportando dados limpos para Parquet...")
conn.execute(
    "COPY precos_tratados TO 'data/precos_combustiveis_2024.parquet' (FORMAT 'PARQUET');"
)
print("✓ Dados exportados para 'data/precos_combustiveis_2024.parquet'")

# Fechar conexão
print("\n[6/6] Fechando conexão com DuckDB...")
conn.close()

print("\n🎉 Pipeline concluído com sucesso!")
