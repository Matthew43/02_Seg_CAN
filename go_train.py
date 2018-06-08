# -*- coding: utf-8 -*-
# @Time    : 2018/6/5 14:52
# @Author  : matthew
# @File    : go_train.py
# @Software: PyCharm

import image_util
import model

# preparing data loading
data_provider = image_util.ImageDataProvider(
    r"/home/matthew/dataset/new-AID/AerialImageDataset/train/images/*.tif", data_suffix='images',
    mask_suffix='gt')

# setup & training
net = model.CA_Net(input_channels=3, n_classes=2)
net.train(data_provider, batch_size=1, output_path='./result_20180606', optimizer='momentum', training_iters=24450,
          epochs=20,
          display_step=100,
          restore=False, write_graph=False)

# #verification
# ...
#
# prediction = net.predict(path, data)
#
# unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
#
# img = util.combine_img_prediction(data, label, prediction)
# util.save_image(img, "prediction.jpg")
