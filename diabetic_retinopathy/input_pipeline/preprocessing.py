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
    # image = tf.image.resize(image, size=(img_height, img_width))

    return image, label


for i in range(3):
    seed = (i, 0)  # tuple of size (2,)


def augment(image, label):
    """Data augmentation"""
    image = apply_randomly(image, augment_contrast)
    image = apply_randomly(image, augment_saturation)
    image = apply_randomly(image, augment_brightness)
    image = apply_randomly(image, augment_flip_left_right)
    image = apply_randomly(image, augment_flip_updowm)
    image = apply_randomly(image, augment_random_crop)

    return image, label


@gin.configurable
def apply_randomly(img, apply_func, p=0.5):
    if tf.random.uniform([]) < p:
        img = apply_func(img)
    else:
        img = img
    return img


def augment_brightness(image):
    img = tf.image.stateless_random_brightness(image, 0.3, seed=seed)
    return img


def augment_contrast(image):
    img = tf.image.stateless_random_contrast(image, 0.2, 1, seed=seed)
    return img


def augment_saturation(image):
    img = tf.image.stateless_random_saturation(image, 0.1, 1.0, seed=seed)
    return img


def augment_flip_left_right(image):
    img = tf.image.stateless_random_flip_left_right(image, seed=seed)
    return img


def augment_flip_updowm(image):
    img = tf.image.stateless_random_flip_up_down(image, seed=seed)
    return img


def augment_random_crop(image):
    h = gin.query_parameter('preprocess.img_height')
    w = gin.query_parameter('preprocess.img_width')
    cropped_h = int(h / 1.3)
    cropped_w = int(w / 1.3)
    img = tf.image.stateless_random_crop(image, (cropped_h, cropped_w, 3), seed=seed)
    img = tf.image.resize(img, [w, h])
    return img