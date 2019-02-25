# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

from tensorflow.contrib import slim
from .mtcnn_model import *


# config to train R-Net (prediction net)
class Config:
    def __init__(self):
        self.image_size = 24
        self.number_of_epochs = 30
        self.number_of_iterations = 5000
        self.batch_size = 384
        self.lr = 0.01
        self.lr_epochs = (6, 14, 20)

        self.pos_ratio = 1
        self.neg_ratio = 3
        self.part_ratio = 1
        self.landmark_ratio = 1

        self.factory = RNet

        # config for database to train net
        self.dbase = None

        # prefix to save trained net
        self.prefix = None


# construct RNet
class RNet:
    def __init__(self, inputs, label=None, bbox_target=None, landmark_target=None, training=True):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='valid'):
            print(inputs.get_shape())
            net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3, 3], stride=1, scope='conv1')
            print(net.get_shape())
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool1', padding='SAME')
            print(net.get_shape())
            net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=1, scope='conv2')
            print(net.get_shape())
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2')
            print(net.get_shape())
            net = slim.conv2d(net, num_outputs=64, kernel_size=[2, 2], stride=1, scope='conv3')
            print(net.get_shape())
            fc_flatten = slim.flatten(net)
            print(fc_flatten.get_shape())
            fc1 = slim.fully_connected(fc_flatten, num_outputs=128, scope='fc1')
            print(fc1.get_shape())

            self.cls_prob = slim.fully_connected(fc1, num_outputs=2, scope='cls_fc', activation_fn=tf.nn.softmax)
            print(self.cls_prob.get_shape())
            self.bbox_pred = slim.fully_connected(fc1, num_outputs=4, scope='bbox_fc', activation_fn=None)
            print(self.bbox_pred.get_shape())
            self.landmark_pred = slim.fully_connected(fc1, num_outputs=10, scope='landmark_fc', activation_fn=None)
            print(self.landmark_pred.get_shape())

            # if train
            if training:
                self.cls_loss = cls_ohem(self.cls_prob, label)
                self.bbox_loss = bbox_ohem(self.bbox_pred, bbox_target, label)
                self.accuracy = cal_accuracy(self.cls_prob, label)
                self.landmark_loss = landmark_ohem(self.landmark_pred, landmark_target, label)
                self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())
