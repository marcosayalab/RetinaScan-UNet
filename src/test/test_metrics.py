import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from metrics import dice_coef, dice_loss

print("Inicializando Test Unitario de Métricas...\n")

# 1. Creamos una "máscara del médico" de juguete (2x2 píxeles)
# Imagina que la fila de arriba es vena (1.0) y la de abajo es fondo (0.0)
y_true_np = np.array([[[[1.0], [1.0]], 
                       [[0.0], [0.0]]]]) 
y_true = tf.constant(y_true_np, dtype=tf.float32)

# 2. ESCENARIO A: La IA hace una predicción perfecta (exactamente igual al médico)
y_pred_perfect = tf.constant(y_true_np, dtype=tf.float32)

# 3. ESCENARIO B: La IA falla estrepitosamente (predice que todo es fondo negro)
y_pred_bad_np = np.zeros_like(y_true_np)
y_pred_bad = tf.constant(y_pred_bad_np, dtype=tf.float32)

# 4. Ejecutamos tus funciones
print("-" * 50)
print("ESCENARIO A: Predicción Perfecta")
# Usamos .numpy() al final solo para extraer el número del tensor y poder imprimirlo bonito
dice_A = dice_coef(y_true, y_pred_perfect).numpy()
loss_A = dice_loss(y_true, y_pred_perfect).numpy()
print(f"Coeficiente DICE : {dice_A:.4f} (Debería ser casi 1.0)")
print(f"Función de Pérdida: {loss_A:.4f} (Debería ser casi 0.0)")

print("-" * 50)
print("ESCENARIO B: Predicción Pésima (Todo negro)")
dice_B = dice_coef(y_true, y_pred_bad).numpy()
loss_B = dice_loss(y_true, y_pred_bad).numpy()
print(f"Coeficiente DICE : {dice_B:.8f} (Debería ser casi 0.0)")
print(f"Función de Pérdida: {loss_B:.8f} (Debería ser casi 1.0)")
print("-" * 50)
