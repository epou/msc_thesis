import tensorflow as tf


def ssim(y_true, y_pred, max_val=1.0):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val))
