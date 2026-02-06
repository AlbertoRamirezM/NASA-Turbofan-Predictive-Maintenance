import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==============================================================================
# 🚀 PASO 3: ENTRENAMIENTO DEL CEREBRO DIGITAL (Random Forest)
# ==============================================================================

FILE_NAME = "train_FD001.txt"

# 1. PREPARACIÓN DE DATOS (Repetimos lo del paso anterior rápido)
columnas = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
sensores = [f's_{i}' for i in range(1, 22)] 
columnas.extend(sensores)

df = pd.read_csv(FILE_NAME, sep=r'\s+', header=None, names=columnas)

# Calculamos RUL
max_life = df.groupby('unit_nr')['time_cycles'].max().reset_index()
max_life.columns = ['unit_nr', 'max_cycle']
df = df.merge(max_life, on='unit_nr', how='left')
df['RUL'] = df['max_cycle'] - df['time_cycles']
df.drop('max_cycle', axis=1, inplace=True)

# Limpiamos basura
cols_a_borrar = ['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']
df_limpio = df.drop(columns=cols_a_borrar)

print("✅ Datos preparados.")

# 2. SEPARACIÓN DE EXAMEN (Train vs Test) 🎓
# ¡OJO! No podemos mezclar filas aleatorias. Tenemos que separar por MOTORES completos.
# Entrenamos con los motores 1 al 80.
# Probamos con los motores 81 al 100 (estos serán "invisibles" para la IA durante el estudio).

motores_totales = df_limpio['unit_nr'].unique()
motores_train = motores_totales[:80] # Los primeros 80
motores_test = motores_totales[80:]  # Los últimos 20

train_df = df_limpio[df_limpio['unit_nr'].isin(motores_train)]
test_df = df_limpio[df_limpio['unit_nr'].isin(motores_test)]

print(f" Motores para estudiar (Train): {len(motores_train)}")
print(f" Motores para el examen (Test): {len(motores_test)}")

# Definimos X (lo que ve la IA) e y (lo que tiene que adivinar)
features = [col for col in df_limpio.columns if col not in ['unit_nr', 'time_cycles', 'RUL']]
target = 'RUL'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# 3. ENTRENAMIENTO (Aquí ocurre la magia) 
print("\n Entrenando Random Forest (esto puede tardar unos segundos)...")
# n_estimators=100 significa que creamos 100 "mini-ingenieros"
modelo = RandomForestRegressor(n_estimators=100, random_state=42) 
modelo.fit(X_train, y_train)

print("✅ ¡Entrenamiento completado!")

# 4. EVALUACIÓN Y PREDICCIÓN 
print("🔮 Realizando predicciones sobre los motores de prueba...")
predicciones = modelo.predict(X_test)

# Calculamos el error (RMSE: Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2 = r2_score(y_test, predicciones)

print("-" * 50)
print(f"📊 RESULTADOS DEL MODELO:")
print(f"   ERROR PROMEDIO (RMSE): {rmse:.2f} ciclos")
print(f"   PRECISIÓN (R2 Score):  {r2:.2f} (1.0 es perfecto)")
print("-" * 50)

# 5. GRÁFICA DE LA VERDAD VS PREDICCIÓN
# Vamos a ver qué tal lo hizo con un motor concreto del set de prueba (ej. Motor 82)
motor_evaluado = 82
indices_motor = test_df[test_df['unit_nr'] == motor_evaluado].index

plt.style.use('ggplot')
plt.figure(figsize=(12, 6))

# Dibujamos la realidad
plt.plot(test_df.loc[indices_motor, 'time_cycles'], 
         y_test.loc[indices_motor], 
         color='black', linewidth=2, label='Realidad (RUL Real)')

# Dibujamos lo que dijo la IA
plt.plot(test_df.loc[indices_motor, 'time_cycles'], 
         predicciones[indices_motor[:len(indices_motor)] - test_df.index[0]], # Ajuste de índice simple
         color='red', linestyle='--', linewidth=2, label='Predicción IA')
# Nota: El índice de 'predicciones' es un array numpy, hay que alinearlo con cuidado,
# pero para simplificar la visualización rápida usaremos un truco directo:

plt.clf() # Limpiamos para hacerlo más fácil de entender

# TRUCO VISUAL: Comparativa general
plt.scatter(y_test, predicciones, alpha=0.1, color='blue')
plt.plot([0, 350], [0, 350], 'r--') # Línea perfecta
plt.xlabel("Vida Real Restante")
plt.ylabel("Vida Predicha por IA")
plt.title(f"PRECISIÓN: Error de {rmse:.0f} ciclos", fontsize=16)
plt.show()