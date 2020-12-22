import functools
import tensorflow as tf

psnr = functools.partial(tf.image.psnr, max_val=1.0)
psnr.__name__ = "psnr"

ssim = functools.partial(tf.image.ssim, max_val=1.0)
ssim.__name__ = "ssim"


__all__ = [psnr, ssim]
