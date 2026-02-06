import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==============================================================================
# 🚀 PASO 4: OPTIMIZACIÓN (RUL CLIPPING)
# ==============================================================================

FILE_NAME = "train_FD001.txt"

# 1. CARGA
columnas = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
sensores = [f's_{i}' for i in range(1, 22)] 
columnas.extend(sensores)
df = pd.read_csv(FILE_NAME, sep=r'\s+', header=None, names=columnas)

# 2. CÁLCULO DE RUL
max_life = df.groupby('unit_nr')['time_cycles'].max().reset_index()
max_life.columns = ['unit_nr', 'max_cycle']
df = df.merge(max_life, on='unit_nr', how='left')
df['RUL'] = df['max_cycle'] - df['time_cycles']
df.drop('max_cycle', axis=1, inplace=True)

# -------------------------------------------------------------------------
# 🔧 EL TRUCO DEL ALMENDRUCO: RUL CLIPPING
# Limitamos el RUL máximo a 125. 
# Hipótesis: A partir de 125 ciclos, el motor está "sano" y los sensores no varían.
LIMIT_RUL = 125
df['RUL_clipped'] = df['RUL'].clip(upper=LIMIT_RUL)
# -------------------------------------------------------------------------

# 3. LIMPIEZA
cols_a_borrar = ['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']
df_limpio = df.drop(columns=cols_a_borrar)

# 4. PREPARACIÓN TRAIN/TEST
motores_train = df_limpio['unit_nr'].unique()[:80]
motores_test = df_limpio['unit_nr'].unique()[80:]

train_df = df_limpio[df_limpio['unit_nr'].isin(motores_train)]
test_df = df_limpio[df_limpio['unit_nr'].isin(motores_test)]

# Ahora el target es 'RUL_clipped', NO el RUL original
features = [col for col in df_limpio.columns if col not in ['unit_nr', 'time_cycles', 'RUL', 'RUL_clipped']]
target = 'RUL_clipped' 

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target] # Ojo: comparamos contra el clipped para ser justos en el entrenamiento

# 5. ENTRENAMIENTO
print(f" Entrenando con RUL limitado a {LIMIT_RUL} ciclos...")
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 6. EVALUACIÓN
predicciones = modelo.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2 = r2_score(y_test, predicciones)

print("-" * 50)
print(f"📊 RESULTADOS MEJORADOS (Clipping a {LIMIT_RUL}):")
print(f"   ERROR PROMEDIO (RMSE): {rmse:.2f} ciclos")
print(f"   PRECISIÓN (R2): {r2:.2f}")
print("-" * 50)

# 7. GRÁFICA COMPARATIVA
plt.style.use('ggplot')
plt.figure(figsize=(10, 6))

# Puntos azules
plt.scatter(y_test, predicciones, alpha=0.1, color='blue', label='Predicciones')

# Línea perfecta
plt.plot([0, 130], [0, 130], 'r--', linewidth=3, label='Predicción Perfecta')

plt.xlabel(f"Vida Real (Limitada a {LIMIT_RUL})")
plt.ylabel("Vida Predicha por IA")
plt.title(f"MODELO OPTIMIZADO: Error bajado a {rmse:.0f} ciclos", fontsize=16)
plt.legend()
plt.show()