import gin
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))

    return image, label


# Data Augmentation

tf.random.set_seed = 100


def random_augment(image, apply_method, prob=0.3):
    if tf.random.uniform([]) <= prob:
        image = apply_method(image)
    else:
        image = image
    return image


def random_flip_left_right(image):
    image = tf.image.flip_up_down(image)
    return image


def random_flip_up_down(image):
    image = tf.image.flip_up_down(image)
    return image


def random_rotate(image):
    random_angle = tf.random.uniform([], seed=5) * 2 * np.pi
    image = tfa.image.rotate(image,
                             angles=random_angle,
                             )
    return image


def random_brightness(image):
    random_value = tf.random.uniform([], minval=0, maxval=0.4, seed=3)
    image = tf.image.adjust_brightness(image,
                                       random_value,
                                       )
    return image


def random_contrast(image):
    random_value = tf.random.uniform([], minval=1, maxval=2, seed=8)
    image = tf.image.adjust_contrast(image,
                                     random_value,
                                     )
    return image


def random_saturation(image):
    random_value = tf.random.uniform([], minval=0.4, maxval=2, seed=8)
    image = tf.image.adjust_saturation(image,
                                       random_value,
                                       )
    return image


def augment(image, label):
    """Data augmentation"""
    image = random_augment(image, random_flip_up_down, prob=0.35)
    image = random_augment(image, random_flip_left_right, prob=0.35)
    image = random_augment(image, random_rotate, prob=0.25)
    image = random_augment(image, random_brightness, prob=0.25)
    image = random_augment(image, random_contrast, prob=0.25)
    image = random_augment(image, random_saturation, prob=0.25)
    return image, label
