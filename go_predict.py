# -*- coding: utf-8 -*-
# @Time    : 2018/6/6 21:24
# @Author  : matthew
# @File    : go_predict.py
# @Software: PyCharm

import model
import utils

net = model.CA_Net(input_channels=3, n_classes=2)
net.predict(model_path=r'E:\experiment\02_Seg_CAN\result_20180606\checkpoint', read_function=utils.read_tiff,
            save_function=utils.save_predict2tiff,
            x_test_path=r'E:\experiment\00_dataset\new-AID\AerialImageDataset\train_back\images\*.tif',
            predict_path=r'E:/experiment/00_dataset/new-AID/AerialImageDataset/train_back/predict_V2')
