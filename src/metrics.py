import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="Custom")
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    numerador = 2. * intersection + smooth
    denominador = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    return numerador / denominador

@keras.saving.register_keras_serializable(package="Custom")
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

@keras.saving.register_keras_serializable(package="Custom")
def bce_dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    bce = tf.keras.losses.binary_crossentropy(y_true_f, y_pred_f)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)