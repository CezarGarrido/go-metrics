#!/usr/bin/env python3
"""
Consultas SQL Analíticas com DuckDB
Dataset: Série Histórica de Preços de Combustíveis - ANP 2024
Autor: Trabalho Final de Data Science
Data: Novembro 2025

Requisito: Mínimo 5 consultas SQL analíticas complexas
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

# Criar diretório para os gráficos
Path("graficos").mkdir(exist_ok=True)

print("=" * 80)
print("CONSULTAS SQL ANALÍTICAS E ANÁLISE EXPLORATÓRIA")
print("Dataset: Preços de Combustíveis ANP 2024")
print("=" * 80)

# Conectar ao banco de dados
conn = duckdb.connect("data/combustiveis.duckdb")

# --- Consulta 1: Evolução Mensal dos Preços Médios por Combustível ---
print("\n[1/5] Consulta 1: Evolução Mensal dos Preços Médios por Combustível")

df_evolucao_mensal = conn.execute("""
    SELECT
        strftime(data_coleta,
        '%Y-%m') as mes,
        produto,
        AVG(valor_venda) as preco_medio
    FROM precos_tratados
    GROUP BY mes, produto
    ORDER BY mes, produto;
""").fetchdf()

plt.figure()
sns.lineplot(data=df_evolucao_mensal, x="mes", y="preco_medio", hue="produto")
plt.title("Evolução Mensal dos Preços Médios por Combustível (2024)")
plt.xlabel("Mês")
plt.ylabel("Preço Médio (R$)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graficos/01_evolucao_mensal_precos.png")
print("  ✓ Gráfico salvo em: graficos/01_evolucao_mensal_precos.png")

# --- Consulta 2: Top 10 Bandeiras com Maiores e Menores Preços Médios de Gasolina ---
print(
    "\n[2/5] Consulta 2: Top 10 Bandeiras com Maiores e Menores Preços Médios de Gasolina"
)

df_bandeiras_gasolina = conn.execute("""
    WITH precos_bandeira AS (
        SELECT
            bandeira,
            AVG(valor_venda) as preco_medio
        FROM precos_tratados
        WHERE produto = 'GASOLINA'
        GROUP BY bandeira
        HAVING COUNT(*) > 1000 -- Considerar apenas bandeiras com mais de 1000 registros
    )
    (SELECT *, 'Top 10 Mais Caras' as tipo FROM precos_bandeira ORDER BY preco_medio DESC LIMIT 10)
    UNION ALL
    (SELECT *, 'Top 10 Mais Baratas' as tipo FROM precos_bandeira ORDER BY preco_medio ASC LIMIT 10);
""").fetchdf()

plt.figure(figsize=(12, 8))
sns.barplot(
    data=df_bandeiras_gasolina, y="bandeira", x="preco_medio", hue="tipo", dodge=False
)
plt.title("Top 10 Bandeiras com Maiores e Menores Preços Médios de Gasolina (2024)")
plt.xlabel("Preço Médio (R$)")
plt.ylabel("Bandeira")
plt.tight_layout()
plt.savefig("graficos/02_top_bandeiras_gasolina.png")
print("  ✓ Gráfico salvo em: graficos/02_top_bandeiras_gasolina.png")

# --- Consulta 3: Variação de Preços (Volatilidade) por Estado ---
print("\n[3/5] Consulta 3: Variação de Preços (Volatilidade) por Estado")

df_volatilidade_estado = conn.execute("""
    SELECT
        estado_sigla,
        STDDEV_POP(valor_venda) as volatilidade,
        AVG(valor_venda) as preco_medio
    FROM precos_tratados
    WHERE produto = 'GASOLINA'
    GROUP BY estado_sigla
    ORDER BY volatilidade DESC;
""").fetchdf()

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_volatilidade_estado,
    x="preco_medio",
    y="volatilidade",
    hue="estado_sigla",
    size="volatilidade",
    sizes=(50, 500),
    legend=False,
)
plt.title("Volatilidade vs. Preço Médio da Gasolina por Estado (2024)")
plt.xlabel("Preço Médio (R$)")
plt.ylabel("Desvio Padrão (Volatilidade)")
# Adicionar labels para os estados
for i, row in df_volatilidade_estado.iterrows():
    plt.text(row["preco_medio"], row["volatilidade"], row["estado_sigla"], fontsize=8)
plt.tight_layout()
plt.savefig("graficos/03_volatilidade_preco_estado.png")
print("  ✓ Gráfico salvo em: graficos/03_volatilidade_preco_estado.png")

# --- Consulta 4: Comparação de Preços entre Capitais e Interior ---
print("\n[4/5] Consulta 4: Comparação de Preços entre Capitais e Interior")

# Lista de capitais brasileiras
capitais = [
    "RIO BRANCO",
    "MACEIO",
    "MACAPA",
    "MANAUS",
    "SALVADOR",
    "FORTALEZA",
    "BRASILIA",
    "VITORIA",
    "GOIANIA",
    "SAO LUIS",
    "CUIABA",
    "CAMPO GRANDE",
    "BELO HORIZONTE",
    "BELEM",
    "JOAO PESSOA",
    "CURITIBA",
    "RECIFE",
    "TERESINA",
    "RIO DE JANEIRO",
    "NATAL",
    "PORTO ALEGRE",
    "PORTO VELHO",
    "BOA VISTA",
    "FLORIANOPOLIS",
    "SAO PAULO",
    "ARACAJU",
    "PALMAS",
]

df_capital_interior = conn.execute(f"""
    SELECT
        CASE WHEN municipio IN {tuple(capitais)} THEN 'Capital' ELSE 'Interior' END as tipo_localizacao,
        produto,
        AVG(valor_venda) as preco_medio
    FROM precos_tratados
    GROUP BY tipo_localizacao, produto
    ORDER BY produto, tipo_localizacao;
""").fetchdf()

plt.figure()
sns.barplot(
    data=df_capital_interior, x="produto", y="preco_medio", hue="tipo_localizacao"
)
plt.title("Preço Médio de Combustíveis: Capital vs. Interior (2024)")
plt.xlabel("Produto")
plt.ylabel("Preço Médio (R$)")
plt.tight_layout()
plt.savefig("graficos/04_preco_capital_vs_interior.png")
print("  ✓ Gráfico salvo em: graficos/04_preco_capital_vs_interior.png")

# --- Consulta 5: Correlação entre Preço da Gasolina e do Etanol ---
print("\n[5/5] Consulta 5: Correlação entre Preço da Gasolina e do Etanol")

df_correlacao = conn.execute("""
    WITH precos_gasolina AS (
        SELECT municipio, estado_sigla, AVG(valor_venda) as preco_gasolina
        FROM precos_tratados
        WHERE produto = 'GASOLINA'
        GROUP BY municipio, estado_sigla
    ),
    precos_etanol AS (
        SELECT municipio, estado_sigla, AVG(valor_venda) as preco_etanol
        FROM precos_tratados
        WHERE produto = 'ETANOL'
        GROUP BY municipio, estado_sigla
    )
    SELECT
        g.municipio,
        g.estado_sigla,
        g.preco_gasolina,
        e.preco_etanol
    FROM precos_gasolina g
    JOIN precos_etanol e ON g.municipio = e.municipio AND g.estado_sigla = e.estado_sigla;
""").fetchdf()

plt.figure()
sns.regplot(
    data=df_correlacao, x="preco_etanol", y="preco_gasolina", line_kws={"color": "red"}
)
plt.title("Correlação entre Preço Médio da Gasolina e do Etanol por Município (2024)")
plt.xlabel("Preço Médio do Etanol (R$)")
plt.ylabel("Preço Médio da Gasolina (R$)")
# Calcular correlação
correlacao = df_correlacao["preco_etanol"].corr(df_correlacao["preco_gasolina"])
plt.text(
    0.1, 0.9, f"Correlação de Pearson: {correlacao:.2f}", transform=plt.gca().transAxes
)
plt.tight_layout()
plt.savefig("graficos/05_correlacao_gasolina_etanol.png")
print(f"  ✓ Gráfico salvo em: graficos/05_correlacao_gasolina_etanol.png")

# Fechar conexão
conn.close()

print(
    "\n🎉 Análise exploratória concluída com sucesso! Gráficos salvos na pasta 'graficos'."
)
