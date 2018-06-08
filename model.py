# -*- coding: utf-8 -*-
# @Time    : 2018/6/1 21:00
# @Author  : matthew
# @File    : model.py
# @Software: PyCharm
import glob
import logging
import os

import numpy as np
import tensorflow as tf

import model_build
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


class CA_Net(object):
    # 构建 实验流图
    def __init__(self, input_channels=3, n_classes=2):
        tf.reset_default_graph()
        self.n_classes = n_classes

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, None, input_channels])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.n_classes])

        logits = model_build.build(self.x, self.n_classes)

        self.loss = self._get_loss(logits)
        self.cross_entropy = tf.reduce_mean(utils.cross_entropy(tf.reshape(self.y, [-1, self.n_classes]),
                                                                tf.reshape(utils.pixel_wise_softmax_2(logits),
                                                                           [-1, self.n_classes])))

        # 计算像素级别的互熵损失
        self.predicter = utils.pixel_wise_softmax_2(logits)
        # 每个类别的正确与否
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        # 计算所有类别的准确率
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_loss(self, logits):
        return tf.reduce_mean(tf.square(logits - self.y))

    def predict(self, model_path, x_test_path, read_function, save_function,predict_path):
        '''
        预测和推断
        :param model_path:
        :param x_test:
        :return: 像素级别每个类别概率图
        '''
        x_paths = glob.glob(x_test_path)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.all_model_checkpoint_paths[-1]:
                self.restore(sess, ckpt.all_model_checkpoint_paths[-1])
                for x in x_paths:
                    # batch = 1
                    x_test = read_function(x)

                    y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_classes))
                    prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy})
                    # save
                    save_function(prediction,x,predict_path)

            else:
                logging.info("Model not find from file: %s" % model_path)

        return

    def save(self, sess, model_path, step):
        '''
        存储模型
        :param sess:
        :param model_path:
        :return:
        '''
        saver = tf.train.Saver(max_to_keep=30)
        checkpoint_path = os.path.join(model_path, 'model{}.ckpt'.format(step))
        save_path = saver.save(sess, checkpoint_path)
        return save_path

    def restore(self, sess, model_path):
        '''
        加载模型
        :param sess:
        :param model_path:
        :return:
        '''
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)

    def _initialize_for_train(self, training_iters, save_path, log_path, prediction_path):
        '''
        训练参数初始化
        '''
        global_step = tf.Variable(0, trainable=False)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        tf.summary.scalar('accuracy', self.accuracy)

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        # init = tf.global_variables_initializer()

        # abs_output_path = os.path.abspath(self.output_path)
        abs_save_path = os.path.abspath(save_path)
        abs_log_path = os.path.abspath(log_path)
        abs_prediction_path = os.path.abspath(prediction_path)

        # if not restore:
        # logging.info("Removing '{:}'".format(abs_output_path))
        # shutil.rmtree(abs_output_path, ignore_errors=True)
        # logging.info("Allocating '{:}'".format())
        # os.makedirs(abs_save_path)
        if not os.path.exists(abs_save_path):
            logging.info("Allocating '{:}'".format(abs_save_path))
            os.makedirs(abs_save_path)

        if not os.path.exists(abs_log_path):
            logging.info("Allocating '{:}'".format(abs_log_path))
            os.makedirs(abs_log_path)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        return init

    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.85)
            momentum = self.opt_kwargs.pop("momentum", 0.9)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=training_iters,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)

            train_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                         **self.opt_kwargs).minimize(self.loss,
                                                                                     global_step=global_step)
        else:
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.0001)
            self.learning_rate_node = tf.Variable(learning_rate)

            train_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                                     **self.opt_kwargs).minimize(self.loss,
                                                                                 global_step=global_step)

        return train_optimizer

    def train(self, data_provider, batch_size, output_path, optimizer="momentum", training_iters=10, epochs=100,
              display_step=1,
              restore=False, write_graph=False, **opt_kwargs):
        """
        训练模型

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        self.verification_batch_size = batch_size
        self.opt_kwargs = opt_kwargs
        self.batch_size = batch_size
        self.optimizer = optimizer

        output_path = output_path
        save_path = os.path.join(output_path, "checkpoint")
        log_path = os.path.join(output_path, "log")
        prediction_path = os.path.join(output_path, "predict")

        if epochs == 0:
            return save_path

        init = self._initialize_for_train(training_iters, save_path, log_path, prediction_path)

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, save_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(save_path)
                if ckpt and ckpt.all_model_checkpoint_paths[-1]:
                    self.restore(sess, ckpt.all_model_checkpoint_paths[-1])

            test_x, test_y = data_provider(self.batch_size)

            self._train_store_prediction(sess, test_x, test_y, "_init", prediction_path)

            summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)
            logging.info("Start optimization")

            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)

                    # Run optimization op (backprop)
                    _, loss, lr = sess.run(
                        (self.optimizer, self.loss, self.learning_rate_node),
                        feed_dict={self.x: batch_x,
                                   self.y: batch_y
                                   })
                    self.lr = lr

                    if step % display_step == 0:
                        self._output_minibatch_stats(sess, summary_writer, step, batch_x,
                                                     batch_y)

                    total_loss += loss

                utils.output_epoch_stats(epoch, total_loss, training_iters, self.lr)
                self._train_store_prediction(sess, test_x, test_y, "epoch_%s" % epoch, prediction_path)

                result_save_path = self.save(sess, save_path, epoch)
            logging.info("Optimization Finished!")
            return result_save_path

    def _train_store_prediction(self, sess, batch_x, batch_y, name, prediction_path):
        loss, prediction = sess.run([self.loss, self.predicter], feed_dict={self.x: batch_x,
                                                                            self.y: batch_y})

        logging.info("Verification error= {:.1f}%, loss= {:.4f}".format(utils.error_rate(prediction, batch_y),
                                                                        loss))

        img = utils.combine_img_prediction(batch_x, batch_y, prediction)
        utils.save_image(img, "%s/%s.jpg" % (prediction_path, name))
        return

    # def _output_epoch_stats(self, epoch, total_loss, training_iters, lr):

    def _output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                        self.loss,
                                                        self.accuracy,
                                                        self.predicter],
                                                       feed_dict={self.x: batch_x,
                                                                  self.y: batch_y})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info(
            "Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step, loss,
                                                                                                           acc,
                                                                                                           utils.error_rate(
                                                                                                               predictions,
                                                                                                               batch_y)))
