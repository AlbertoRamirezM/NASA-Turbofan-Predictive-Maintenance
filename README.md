NASA Turbofan Predictive Maintenance

Predicting the remaining useful life (RUL) of turbofan jet engines from sensor data, using the public NASA C-MAPSS dataset (FD001 subset).

Dataset

train_FD001.txt: sensor trajectories for several turbofan engines up to failure. Each row is one flight cycle for a given engine, with 26 columns: engine unit number, flight cycle, 3 operational settings, and 21 sensors.

Pipeline

The project is organized into 5 sequential steps:

1_analisis_inicial.py: initial exploration. Plots the evolution of three key sensors (compressor outlet temperature, turbine outlet temperature, static inlet pressure) for a given engine, with a smoothed trend line and the failure point marked.

2_preparacion_datos.py: target construction. Computes RUL for each record (the engine’s maximum cycle minus the current cycle), drops operational settings and sensors with no informative variation in this subset, plots the RUL curve for several engines, and checks the correlation of each remaining sensor with RUL.

3_entrenamiento_modelo.py: baseline model. Trains a Random Forest on the 15 informative sensors, splitting by engine (not by row) between training and test sets, and evaluates with RMSE and R².

4_mejora_modelo.py: model iteration. Clips RUL at a maximum of 125 cycles, based on the idea that above that threshold the engine is healthy and distinguishing finer degrees of “very healthy” adds little practical maintenance value.

5_demo_final.py: control panel demo. Trains the final model on all available data and simulates, in the console, a real-time maintenance alert system for two example engines, with a traffic-light status (OK / caution / critical) based on the predicted RUL.


Next steps

Evaluate the final demo on engines unseen during training, instead of on the same training data.
Compare the RUL-clipping improvement (step 4) against the true, unclipped RUL, to confirm the accuracy gain isn’t overstated by measuring only within the clipped range.
Explore time-derived features (rolling averages, trends) instead of using only the raw sensor value at the current cycle.
