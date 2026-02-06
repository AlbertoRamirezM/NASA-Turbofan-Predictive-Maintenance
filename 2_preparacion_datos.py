import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 🚀 PASO 2: INGENIERÍA DE DATOS (Cálculo del RUL y Limpieza)
# ==============================================================================

FILE_NAME = "train_FD001.txt"

# 1. CARGA DE DATOS (Igual que antes)
columnas = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
sensores = [f's_{i}' for i in range(1, 22)] 
columnas.extend(sensores)

df = pd.read_csv(FILE_NAME, sep=r'\s+', header=None, names=columnas)

print(f"📊 Datos originales cargados: {df.shape}")

# 2. CÁLCULO DEL RUL (Remaining Useful Life) 
# Lógica: Agrupamos por motor -> Buscamos cuál fue su último ciclo -> Restamos el ciclo actual
# Esto crea una cuenta atrás perfecta.

# Buscamos el ciclo máximo para cada motor
max_life = df.groupby('unit_nr')['time_cycles'].max().reset_index()
max_life.columns = ['unit_nr', 'max_cycle']

# Lo pegamos al dataframe original
df = df.merge(max_life, on='unit_nr', how='left')

# Calculamos la resta: Vida Total - Vida Actual
df['RUL'] = df['max_cycle'] - df['time_cycles']

# Borramos la columna auxiliar 'max_cycle' que ya no sirve
df.drop('max_cycle', axis=1, inplace=True)

print("✅ Columna RUL calculada (Cuenta atrás lista).")

# 3. LIMPIEZA DE SENSORES INÚTILES 🧹
# En este dataset específico de la NASA, hay sensores que NO cambian nunca.
# Tener datos que valen siempre lo mismo confunde a la IA. Los borramos.
# (Sensores: 1, 5, 10, 16, 18, 19 son constantes en FD001)

cols_a_borrar = ['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']
df_limpio = df.drop(columns=cols_a_borrar)

print(f" Columnas inútiles borradas. Dimensiones finales: {df_limpio.shape}")
print("-" * 50)

# 4. VISUALIZACIÓN DE LA CUENTA ATRÁS
# Vamos a ver cómo baja la vida de los primeros 4 motores.

plt.style.use('ggplot')
plt.figure(figsize=(12, 6))

for motor_id in [1, 2, 3, 4]:
    datos_motor = df_limpio[df_limpio['unit_nr'] == motor_id]
    plt.plot(datos_motor['time_cycles'], datos_motor['RUL'], linewidth=3, label=f'Motor {motor_id}')

plt.xlabel("Ciclos de Vuelo (Tiempo)", fontsize=12)
plt.ylabel("RUL (Vida Restante)", fontsize=12)
plt.title("LA CUENTA ATRÁS: Evolución del RUL en 4 motores", fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

# 5. VERIFICACIÓN DE CORRELACIÓN
# ¿Qué sensor se parece más a la cuenta atrás?
# Esto nos dice qué sensor es el mejor "chivato" del fallo.
plt.figure(figsize=(10, 8))
heatmap_data = df_limpio.corr() # Calculamos correlación matemática
# Solo nos interesa la columna RUL ordenada
rul_corr = heatmap_data[['RUL']].sort_values(by='RUL', ascending=False)

sns.heatmap(rul_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("MAPA DE CALOR: ¿Qué sensores avisan mejor del fallo?", fontsize=14)
plt.show()