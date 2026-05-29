import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    numerador = 2. * intersection + smooth
    denominador = K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    return numerador / denominador

@keras.saving.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

@keras.saving.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    bce = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    return bce + dice_loss(y_true, y_pred)