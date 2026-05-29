"""
Módulo de Métricas para Segmentación Médica.
Implementa el Coeficiente DICE utilizando el backend de Keras para 
garantizar la compatibilidad con el grafo computacional de TensorFlow.
"""

import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Calcula el Coeficiente DICE (equivalente al F1-Score a nivel de píxel) 
    para evaluar el solapamiento entre la predicción y el Ground Truth.
    
    Parámetros:
    -----------
    y_true : tf.Tensor
        Máscara real etiquetada por el especialista (1 = vena, 0 = fondo).
    y_pred : tf.Tensor
        Matriz de probabilidades predicha por la U-Net.
    smooth : float, opcional
        Factor de suavizado para prevenir el error 'ZeroDivisionError' en 
        parches donde no existen vasos sanguíneos (por defecto 1e-6).
        
    Retorna:
    --------
    tf.Tensor
        Un tensor escalar con el valor del coeficiente (entre 0.0 y 1.0).
    """
    # 1. Aplanamos las matrices multidimensionales a vectores 1D
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    # 2. Calculamos la intersección matemática (coincidencias)
    intersection = K.sum(y_true_f * y_pred_f)
    
    # 3. Aplicamos la ecuación del Coeficiente DICE
    numerador = 2. * intersection + smooth
    denominador = K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    
    return numerador / denominador


def dice_loss(y_true, y_pred):
    """
    Función de pérdida (Loss Function) a optimizar durante el entrenamiento.
    Dado que los optimizadores buscan minimizar una métrica, invertimos 
    el DICE restándolo de 1.0. Un DICE perfecto (1.0) dará un error de 0.0.
    
    Parámetros:
    -----------
    y_true : tf.Tensor
        Máscara real.
    y_pred : tf.Tensor
        Predicción de la red neuronal.
        
    Retorna:
    --------
    tf.Tensor
        Un tensor escalar con el valor de la pérdida.
    """
    return 1.0 - dice_coef(y_true, y_pred)