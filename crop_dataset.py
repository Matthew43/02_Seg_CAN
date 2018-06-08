# -*- coding: utf-8 -*-
# @Time    : 2018/6/5 16:51
# @Author  : matthew
# @File    : crop_dataset.py
# @Software: PyCharm
import glob

import tifffile as tiff
import os


def fix_dataset(path):
    all_files = glob.glob(path)

    size = 500
    for i in range(100):
        image_name = all_files[i]
        label_name = image_name.replace('images', 'gt')
        # print(image_name)
        image = tiff.imread(image_name)
        # print(image)
        label = tiff.imread(label_name)
        for x in range(0, 5000, size):
            for y in range(0, 5000, size):
                # print(x, y)
                new_image_name = image_name.replace('.tif', '_%d_%d_%d.tif' % (x, y, size))
                new_image_name = new_image_name.replace('NEW2-AerialImageDataset', 'new-AID')

                new_label_name = label_name.replace('.tif', '_%d_%d_%d.tif' % (x, y, size))
                new_label_name = new_label_name.replace('NEW2-AerialImageDataset', 'new-AID')

                tiff.imsave(new_image_name, image[x:x + size, y:y + size, :])
                tiff.imsave(new_label_name, label[x:x + size, y:y + size])
        print('processing number %d' % (i))

    size = 300
    for i in range(100, 150):
        image_name = all_files[i]
        label_name = image_name.replace('images', 'gt')
        # print(image_name)
        image = tiff.imread(image_name)
        # print(image)
        label = tiff.imread(label_name)
        for x in range(0, 5000, size):
            for y in range(0, 5000, size):
                # print(x, y)
                new_image_name = image_name.replace('.tif', '_%d_%d_%d.tif' % (x, y, size))
                new_image_name = new_image_name.replace('NEW2-AerialImageDataset', 'new-AID')

                new_label_name = label_name.replace('.tif', '_%d_%d_%d.tif' % (x, y, size))
                new_label_name = new_label_name.replace('NEW2-AerialImageDataset', 'new-AID')

                tiff.imsave(new_image_name, image[x:x + size, y:y + size, :])
                tiff.imsave(new_label_name, label[x:x + size, y:y + size])
        print('processing number %d' % (i))

        ## 测试
    size = 500
    for i in range(150, 168):
        image_name = all_files[i]
        label_name = image_name.replace('images', 'gt')
        # print(image_name)
        image = tiff.imread(image_name)
        # print(image)
        label = tiff.imread(label_name)
        for x in range(0, 5000, size):
            for y in range(0, 5000, size):
                # print(x, y)
                new_image_name = image_name.replace('.tif', '_%d_%d_%d.tif' % (x, y, size))
                new_image_name = new_image_name.replace('NEW2-AerialImageDataset', 'new-AID')
                new_image_name = new_image_name.replace('train', 'test')

                new_label_name = label_name.replace('.tif', '_%d_%d_%d.tif' % (x, y, size))
                new_label_name = new_label_name.replace('NEW2-AerialImageDataset', 'new-AID')
                new_label_name = new_label_name.replace('train', 'test')

                tiff.imsave(new_image_name, image[x:x + size, y:y + size, :])
                tiff.imsave(new_label_name, label[x:x + size, y:y + size])
        print('processing number %d' % (i))

    ## 测试
    size = 300
    for i in range(168, 180):
        image_name = all_files[i]
        label_name = image_name.replace('images', 'gt')
        # print(image_name)
        image = tiff.imread(image_name)
        # print(image)
        label = tiff.imread(label_name)
        for x in range(0, 5000, size):
            for y in range(0, 5000, size):
                # print(x, y)
                new_image_name = image_name.replace('.tif', '_%d_%d_%d.tif' % (x, y, size))
                new_image_name = new_image_name.replace('NEW2-AerialImageDataset', 'new-AID')
                new_image_name = new_image_name.replace('train', 'test')

                new_label_name = label_name.replace('.tif', '_%d_%d_%d.tif' % (x, y, size))
                new_label_name = new_label_name.replace('NEW2-AerialImageDataset', 'new-AID')
                new_label_name = new_label_name.replace('train', 'test')

                tiff.imsave(new_image_name, image[x:x + size, y:y + size, :])
                tiff.imsave(new_label_name, label[x:x + size, y:y + size])
        print('processing number %d' % (i))


if __name__ == '__main__':
    os.makedirs('/home/matthew/dataset/new-AID/AerialImageDataset/train/images')
    os.makedirs('/home/matthew/dataset/new-AID/AerialImageDataset/train/gt')
    os.makedirs('/home/matthew/dataset/new-AID/AerialImageDataset/test/images')
    os.makedirs('/home/matthew/dataset/new-AID/AerialImageDataset/test/gt')

    fix_dataset(r'/home/matthew/dataset/NEW2-AerialImageDataset/AerialImageDataset/train/images/*tif')
