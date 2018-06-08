# -*- coding: utf-8 -*-
# @Time    : 2018/6/5 11:42
# @Author  : matthew
# @File    : utils.py
# @Software: PyCharm
import logging
import os

import numpy as np
import tensorflow as tf
import tifffile as tiff
from PIL import Image


# def crop_to_shape(data, shape):
#     """
#     Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
#
#     :param data: the array to crop
#     :param shape: the target shape
#     """
#     offset0 = (data.shape[1] - shape[1]) // 2
#     offset1 = (data.shape[2] - shape[2]) // 2
#     return data[:, offset0:(-offset0), offset1:(-offset1)]


# 像素级别的softmax
def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)

    # 通道维度上的加和
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)

    # a1 = tf.tile(a, [2, 2]) 表示把a的第一个维度复制两次，第二个维度复制2次。
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map, tensor_sum_exp)


def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, (1, 1, 3))

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image

    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor

    :returns img: the concatenated rgb image
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate((to_rgb(data.reshape(-1, ny, ch)),
                          to_rgb(gt[..., 1].reshape(-1, ny, 1)),
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    return img


def save_image(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300, 300], quality=90)


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))


def output_epoch_stats(epoch, total_loss, training_iters, lr):
    logging.info(
        "Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))


def save_predict2tiff(predictions, images_input, predict_path):
    # images = np.sum(predictions, axis=-1, keepdims=False, dtype=np.float32).round() * 255.0
    images = np.array(predictions[0, :, :, 1]).round() * 255.0
    images = images.astype(np.uint8)
    os.makedirs(predict_path, exist_ok=True)
    image_name = images_input.split('\\')[-1].replace('.tif', '_predict.tif')
    image_path = os.path.join(predict_path, image_name)
    print(image_path)
    tiff.imsave(image_path, images[:, :])


def read_tiff(path):
    image = np.array(Image.open(path), dtype=np.float32)
    X = np.zeros((1, image.shape[0], image.shape[1], image.shape[2]))
    X[0] = image
    return X
