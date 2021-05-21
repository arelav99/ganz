import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.metrics import roc_curve, auc


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred)))


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))


def mae_batch(preds, real):
    return K.mean(K.abs(preds - real), axis=[-1, -2, -3])


def mse_batch(preds, real):
    return K.mean(K.square(K.abs(preds - real)), axis=[-1, -2, -3])


def calculate_auc(preds, real):
    fpr, tpr, _ = roc_curve(real, preds)
    return auc(fpr, tpr)
