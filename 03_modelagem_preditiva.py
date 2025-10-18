#!/usr/bin/env python3
"""
Modelagem Preditiva com Machine Learning
Dataset: Série Histórica de Preços de Combustíveis - ANP 2024
Autor: Trabalho Final de Data Science
Data: Novembro 2025

Objetivo: Prever o preço de venda de combustíveis com base em features geográficas,
temporais e características do estabelecimento.

Modelos:
- Baseline: Regressão Linear
- Modelos Complexos: Random Forest, Gradient Boosting (XGBoost)
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MODELAGEM PREDITIVA - MACHINE LEARNING")
print("Dataset: Preços de Combustíveis ANP 2024")
print("=" * 80)

# Conectar ao banco de dados
conn = duckdb.connect("data/combustiveis.duckdb")

# --- Feature Engineering ---
print("\n[1/5] Feature Engineering...")

# Extrair features do banco de dados
df = conn.execute("""
    SELECT
        regiao_sigla,
        estado_sigla,
        municipio,
        produto,
        bandeira,
        EXTRACT(MONTH FROM data_coleta) as mes,
        EXTRACT(DAY FROM data_coleta) as dia,
        valor_venda
    FROM precos_tratados
    WHERE valor_venda IS NOT NULL
        AND bandeira IS NOT NULL
        AND bandeira != ''
    LIMIT 100000  -- Limitar para acelerar o treinamento
""").fetchdf()

print(f"  ✓ {len(df)} registros carregados")

# Codificar variáveis categóricas
label_encoders = {}
categorical_cols = ['regiao_sigla', 'estado_sigla', 'municipio', 'produto', 'bandeira']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Separar features e target
X = df[['regiao_sigla_encoded', 'estado_sigla_encoded', 'municipio_encoded', 
        'produto_encoded', 'bandeira_encoded', 'mes', 'dia']]
y = df['valor_venda']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  ✓ Dados divididos em treino ({len(X_train)} registros) e teste ({len(X_test)} registros)")

# --- Modelagem ---
print("\n[2/5] Treinando modelos...")

models = {
    "Regressão Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

for name, model in models.items():
    print(f"  > Treinando {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    }
    print(f"    ✓ {name} treinado. R²: {r2:.4f}")

# --- Avaliação dos Modelos ---
print("\n[3/5] Avaliando modelos...")

df_results = pd.DataFrame(results).T
print(df_results)

# --- Análise de Features Importantes (Random Forest) ---
print("\n[4/5] Analisando features importantes (Random Forest)...")

rf_model = models["Random Forest"]
feature_importances = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print(feature_importances)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances, x="importance", y="feature")
plt.title("Importância das Features - Random Forest")
plt.tight_layout()
plt.savefig("graficos/06_feature_importance.png")
print("  ✓ Gráfico salvo em: graficos/06_feature_importance.png")

# --- Exemplo de Predição ---
print("\n[5/5] Exemplo de predição...")

# Pegar um exemplo do dataset de teste
sample = X_test.iloc[0].to_dict()
actual_price = y_test.iloc[0]

# Decodificar para visualização
sample_decoded = {}
for col, le in label_encoders.items():
    encoded_col = col + '_encoded'
    if encoded_col in sample:
        sample_decoded[col] = le.inverse_transform([int(sample[encoded_col])])[0]

print("  > Dados de entrada (decodificados):")
print(f"    - Região: {sample_decoded.get('regiao_sigla')}")
print(f"    - Estado: {sample_decoded.get('estado_sigla')}")
print(f"    - Município: {sample_decoded.get('municipio')}")
print(f"    - Produto: {sample_decoded.get('produto')}")
print(f"    - Bandeira: {sample_decoded.get('bandeira')}")
print(f"    - Mês: {int(sample['mes'])}")
print(f"    - Dia: {int(sample['dia'])}")

# Fazer predição com o melhor modelo (Random Forest)
predicted_price = rf_model.predict([list(sample.values())])[0]

print(f"\n  > Preço real: R$ {actual_price:.2f}")
print(f"  > Preço previsto (Random Forest): R$ {predicted_price:.2f}")

# Fechar conexão
conn.close()

print("\n🎉 Modelagem preditiva concluída com sucesso!")
