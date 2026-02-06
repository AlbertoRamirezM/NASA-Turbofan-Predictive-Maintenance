import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 🚀 PROYECTO NASA: PREDICTOR DE FALLOS (V1.3 - MODO LOCAL)
# ==============================================================================

FILE_NAME = "train_FD001.txt"

# 1. VERIFICACIÓN DE ARCHIVO
print(f"📂 Buscando archivo '{FILE_NAME}' en la carpeta actual...")

# Truco: Imprimimos la ruta exacta donde Python está buscando, para que sepas dónde ponerlo
ruta_actual = os.getcwd()
ruta_completa = os.path.join(ruta_actual, FILE_NAME)

if not os.path.exists(FILE_NAME):
    print("\n❌ ERROR: NO SE ENCUENTRA EL ARCHIVO.")
    print(f"⚠️  Python está buscando el archivo aquí:\n    👉 {ruta_completa}")
    print("\nSOLUCIÓN:")
    print("1. Descarga el ZIP de: https://ti.arc.nasa.gov/c/6/")
    print("2. Descomprímelo.")
    print(f"3. Copia 'train_FD001.txt' y pégalo en: {ruta_actual}")
    exit()
else:
    print("✅ ¡Archivo encontrado! Iniciando análisis forense del motor...")

# 2. CARGAR DATOS
columnas = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
sensores = [f's_{i}' for i in range(1, 22)] 
columnas.extend(sensores)

try:
    df = pd.read_csv(FILE_NAME, sep=r'\s+', header=None, names=columnas)
except Exception as e:
    print(f"❌ Error leyendo el archivo (¿está corrupto?): {e}")
    exit()

# 3. VISUALIZACIÓN
print("📊 Generando gráficas del Motor 1...")

plt.style.use('ggplot')
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Filtramos solo el Motor 1
motor_1 = df[df['unit_nr'] == 1]
vida_util = motor_1['time_cycles'].max()

# Definimos sensores críticos (Compresor y Turbina)
sensores_clave = ['s_2', 's_4', 's_11']
titulos = [
    'Temperatura Salida Compresor (s_2)', 
    'Temperatura Salida Turbina (s_4)', 
    'Presión Estática Entrada (s_11)'
]

for i, col in enumerate(sensores_clave):
    ax = axes[i]
    # Datos crudos (azul)
    ax.plot(motor_1['time_cycles'], motor_1[col], color='steelblue', alpha=0.6, label='Lectura Sensor')
    # Tendencia suavizada (rojo)
    ax.plot(motor_1['time_cycles'], motor_1[col].rolling(15).mean(), color='crimson', linewidth=2, label='Tendencia')
    
    ax.set_ylabel(titulos[i], fontsize=10)
    
    # Marca de fallo
    ax.axvline(x=vida_util, color='black', linestyle=':', alpha=0.6)
    if i == 0:
        ax.set_title(f"HISTORIA CLÍNICA MOTOR 1 (Fallo en ciclo {vida_util})", fontsize=14)
        ax.text(vida_util - 50, motor_1[col].max(), '❌ PUNTO DE FALLO', color='red', fontweight='bold')
    
    ax.legend(loc='upper left')

plt.xlabel("Ciclos de Vuelo")
plt.tight_layout()
plt.show()

print("\n FASE 1 COMPLETADA. Si ves las gráficas, avísame para pasar a la IA.")