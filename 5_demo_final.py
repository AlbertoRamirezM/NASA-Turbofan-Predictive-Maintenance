import pandas as pd
import numpy as np
import joblib  # Para guardar/cargar el cerebro de la IA
from sklearn.ensemble import RandomForestRegressor
import time

# ==============================================================================
# 🚀 FASE FINAL: LA APLICACIÓN DE MANTENIMIENTO PREDICTIVO
# ==============================================================================

FILE_NAME = "train_FD001.txt"

# --- PASO 1: RE-ENTRENAR Y GUARDAR EL MODELO (Para tenerlo listo) ---
print("  Configurando sistema...")

# 1. Cargar y Preparar (Resumido)
columnas = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
df = pd.read_csv(FILE_NAME, sep=r'\s+', header=None, names=columnas)

# Calcular RUL y Clipping
max_life = df.groupby('unit_nr')['time_cycles'].max().reset_index()
max_life.columns = ['unit_nr', 'max_cycle']
df = df.merge(max_life, on='unit_nr', how='left')
df['RUL'] = df['max_cycle'] - df['time_cycles']
df['RUL_clipped'] = df['RUL'].clip(upper=125)

# Limpiar columnas inútiles
cols_borrar = ['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']
df_limpio = df.drop(columns=cols_borrar)

# Entrenar con TODOS los datos disponibles (Ya sabemos que funciona bien)
features = [c for c in df_limpio.columns if c not in ['unit_nr', 'time_cycles', 'RUL', 'RUL_clipped']]
X = df_limpio[features]
y = df_limpio['RUL_clipped']

print(" Entrenando modelo final...")
modelo_final = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_final.fit(X, y)

# GUARDAR EL MODELO EN UN ARCHIVO .PKL
# Esto permite usar la IA luego sin tener que volver a entrenar
joblib.dump(modelo_final, 'cerebro_motor_nasa.pkl')
print(" Modelo guardado como 'cerebro_motor_nasa.pkl'")
print("-" * 60)

# --- PASO 2: SIMULACIÓN EN TIEMPO REAL  ---

def panel_de_control(motor_id):
    print(f"\n CONECTANDO CON MOTOR #{motor_id} EN VUELO...")
    time.sleep(1)
    
    # Extraemos los datos de ese motor
    datos_motor = df_limpio[df_limpio['unit_nr'] == motor_id]
    
    # Tomamos los últimos 5 ciclos de su vida
    # (NOTA: Aunque el motor 81 duró mucho, aquí estamos viendo SUS ÚLTIMOS MOMENTOS)
    ciclos_finales = datos_motor.tail(5)
    
    print(f"{'CICLO':<10} | {'SENSOR PRESION (s_11)':<25} | {'PREDICCIÓN RUL':<20} | {'ESTADO'}")
    print("-" * 80)
    
    for i, fila in ciclos_finales.iterrows():
        # CORRECCIÓN: Convertimos la fila a DataFrame para que tenga nombres de columnas
        # Así la IA no se queja del warning
        datos_sensor = pd.DataFrame([fila[features]], columns=features)
        
        # LA IA PREDICE
        rul_predicho = modelo_final.predict(datos_sensor)[0]
        
        # Lógica de Semáforo
        if rul_predicho > 50:
            estado = "🟢 OK"
        elif rul_predicho > 20:
            estado = "🟡 PRECAUCIÓN (Planificar Revisión)"
        else:
            estado = "🔴 PELIGRO CRÍTICO (Aterrizar)"
            
        print(f"Vuelo {int(fila['time_cycles']):<4} | {fila['s_11']:<25.2f} | {rul_predicho:<20.1f} | {estado}")
        time.sleep(0.8)

# PROBAMOS LA DEMO CON UN MOTOR QUE FALLÓ PRONTO (El Motor 34)
panel_de_control(motor_id=34)

# PROBAMOS CON UNO QUE DURÓ MUCHO (El Motor 81)
panel_de_control(motor_id=81)