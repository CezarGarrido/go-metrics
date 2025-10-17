#!/usr/bin/env python3
"""
Etapa 2: Integração e Limpeza de Dados
Projeto Final - Ciência de Dados: Métricas de Código Go
Dataset: Métricas de Código Go para Predição de Refatoração
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import duckdb

warnings.filterwarnings("ignore")

# Configuração de estilo profissional
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300

# Carregar dados usando DuckDB
df = duckdb.query("""
    SELECT * 
    FROM read_csv_auto('go_metrics.csv')
""").df()

print(f"Dataset carregado: {df.shape[0]} entidades, {df.shape[1]} features")

# Remover duplicatas usando DuckDB
initial_count = len(df)
df = duckdb.query("""
    SELECT DISTINCT *
    FROM df
""").df()
print(f"{initial_count - len(df)} duplicatas removidas")

# Definir colunas de métricas
metrics_cols = ["loc", "nom", "nof", "wmc", "cbo", "rfc", "lcom"]

# Função Auxiliar para Plotagem
def safe_kde_plot(data, ax, color="#A23B72", label="Densidade"):
    """Plot KDE seguro que lida com dados de baixa variância"""
    try:
        # Remove NaN e inf
        clean_data = data.dropna()
        if len(clean_data) < 2:
            return

        # Verifica se há variância suficiente
        if clean_data.std() > 1e-10:  # Threshold mínimo de variância
            # Usa o KDE do seaborn que é mais robusto
            sns.kdeplot(clean_data, ax=ax, color=color, linewidth=2.5, label=label)
        else:
            # Para dados com pouca variância, plota uma linha vertical
            ax.axvline(clean_data.iloc[0], color=color, linewidth=2.5, label=label)
    except Exception as e:
        print(f"Aviso: Não foi possível plotar KDE para {data.name}: {e}")

# Análise de Outliers usando DuckDB
print("\n=== ANÁLISE DE OUTLIERS ===")

# Criar uma conexão DuckDB para análise
conn = duckdb.connect()

# Registrar o DataFrame no DuckDB
conn.register('df', df)

# Estatísticas descritivas usando DuckDB - abordagem individual
print("Estatísticas Descritivas das Métricas:")

for col in metrics_cols:
    stats_query = f"""
    SELECT 
        '{col}' as metric,
        COUNT(*) as count,
        AVG({col}) as mean,
        STDDEV({col}) as std,
        MIN({col}) as min,
        MAX({col}) as max,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) as q1,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {col}) as median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) as q3
    FROM df
    """
    
    result = conn.execute(stats_query).df()
    print(f"\n{col.upper()}:")
    print(f"  Count: {result['count'].iloc[0]}")
    print(f"  Mean: {result['mean'].iloc[0]:.2f}")
    print(f"  Std: {result['std'].iloc[0]:.2f}")
    print(f"  Min: {result['min'].iloc[0]}")
    print(f"  Max: {result['max'].iloc[0]}")
    print(f"  Q1: {result['q1'].iloc[0]:.2f}")
    print(f"  Median: {result['median'].iloc[0]:.2f}")
    print(f"  Q3: {result['q3'].iloc[0]:.2f}")

# Detecção de outliers usando regra do IQR
print("\nDetecção de Outliers (Regra IQR):")

outliers_results = []
for col in metrics_cols:
    # Primeiro calcular Q1 e Q3
    quartiles_query = f"""
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) as q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) as q3
    FROM df
    """
    quartiles = conn.execute(quartiles_query).df().iloc[0]
    q1, q3 = quartiles['q1'], quartiles['q3']
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Contar outliers
    outliers_query = f"""
    SELECT 
        COUNT(*) as total_count,
        SUM(CASE WHEN {col} < {lower_bound} OR {col} > {upper_bound} THEN 1 ELSE 0 END) as outlier_count
    FROM df
    """
    
    outliers_info = conn.execute(outliers_query).df().iloc[0]
    total_count = outliers_info['total_count']
    outlier_count = outliers_info['outlier_count']
    outlier_percentage = (outlier_count / total_count) * 100 if total_count > 0 else 0
    
    outliers_results.append({
        'metric': col,
        'total_observations': total_count,
        'outlier_count': outlier_count,
        'outlier_percentage': outlier_percentage,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })
    
    print(f"{col.upper()}: {outlier_count}/{total_count} outliers ({outlier_percentage:.2f}%)")

# Visualização de outliers
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(metrics_cols):
    if i < len(axes):
        # Usar DuckDB para obter dados da métrica específica
        data = conn.execute(f"SELECT {col} FROM df").df()[col]
        
        # Boxplot
        sns.boxplot(y=data, ax=axes[i], color="#A23B72")
        axes[i].set_title(f'Outliers - {col.upper()}')
        axes[i].set_ylabel('Valor')

# Remover eixos extras
for j in range(len(metrics_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Análise de Outliers nas Métricas de Código', y=1.02, fontsize=16)
plt.show()

# Análise de correlação usando DuckDB
print("\n=== ANÁLISE DE CORRELAÇÃO ===")

# Criar matriz de correlação manualmente
correlation_matrix = np.eye(len(metrics_cols))

print("Matriz de Correlação:")
print("       " + " ".join(f"{m:>6}" for m in metrics_cols))

for i, col1 in enumerate(metrics_cols):
    row = [f"{col1:>6}"]
    for j, col2 in enumerate(metrics_cols):
        if i == j:
            correlation_matrix[i, j] = 1.0
            row.append(" 1.000")
        else:
            corr_query = f"SELECT CORR({col1}, {col2}) as correlation FROM df"
            corr_result = conn.execute(corr_query).df().iloc[0]['correlation']
            correlation_matrix[i, j] = corr_result if corr_result is not None else 0.0
            row.append(f" {corr_result:.3f}" if corr_result is not None else "   NaN")
    print(" ".join(row))

# Visualização da matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            xticklabels=metrics_cols, 
            yticklabels=metrics_cols,
            annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': .8})
plt.title('Matriz de Correlação - Métricas de Código Go')
plt.tight_layout()
plt.show()

# Análise de distribuição usando DuckDB
print("\n=== ANÁLISE DE DISTRIBUIÇÃO ===")

print("Análise de Distribuição:")
for col in metrics_cols:
    distribution_query = f"""
    SELECT 
        COUNT(*) as count,
        AVG({col}) as mean,
        STDDEV({col}) as std,
        MIN({col}) as min,
        MAX({col}) as max
    FROM df
    """
    
    dist_result = conn.execute(distribution_query).df().iloc[0]
    
    # Calcular skewness e kurtosis manualmente usando pandas
    data = conn.execute(f"SELECT {col} FROM df").df()[col]
    skewness = data.skew()
    kurtosis = data.kurtosis()
    
    print(f"\n{col.upper()}:")
    print(f"  Count: {dist_result['count']}")
    print(f"  Mean: {dist_result['mean']:.2f}")
    print(f"  Std: {dist_result['std']:.2f}")
    print(f"  Min: {dist_result['min']}")
    print(f"  Max: {dist_result['max']}")
    print(f"  Skewness: {skewness:.2f}")
    print(f"  Kurtosis: {kurtosis:.2f}")

# Visualização de distribuições
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(metrics_cols):
    if i < len(axes):
        # Obter dados usando DuckDB
        data = conn.execute(f"SELECT {col} FROM df").df()[col]
        
        # Histograma com KDE
        sns.histplot(data, ax=axes[i], kde=True, color="#A23B72", alpha=0.7)
        axes[i].set_title(f'Distribuição - {col.upper()}')
        axes[i].set_xlabel('Valor')
        axes[i].set_ylabel('Frequência')

# Remover eixos extras
for j in range(len(metrics_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Distribuição das Métricas de Código', y=1.02, fontsize=16)
plt.show()

# Análise adicional: Valores nulos
print("\n=== ANÁLISE DE VALORES NULOS ===")
nulls_query = "SELECT "
null_checks = []
for col in df.columns:
    null_checks.append(f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as {col}_nulls")
nulls_query += ", ".join(null_checks) + " FROM df"

null_results = conn.execute(nulls_query).df()
print("Valores nulos por coluna:")
for col in df.columns:
    null_count = null_results[f"{col}_nulls"].iloc[0]
    if null_count > 0:
        print(f"  {col}: {null_count} valores nulos")

# Fechar conexão
conn.close()

print("\n=== ANÁLISE CONCLUÍDA ===")
print(f"Dataset final: {len(df)} entidades, {len(df.columns)} features")
print(f"Métricas analisadas: {metrics_cols}")