# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

from tensorflow.contrib import slim
from .mtcnn import *


# config to train P-Net (prediction net)
class Config:
    def __init__(self):
        self.image_size = 12
        self.number_of_epochs = 30
        self.number_of_iterations = 5000
        self.batch_size = 384
        self.lr = 0.01
        self.lr_epochs = (6, 14, 20)

        self.pos_ratio = 1
        self.neg_ratio = 3
        self.part_ratio = 1
        self.landmark_ratio = 1

        self.cls_loss_factor = 1.0
        self.bbox_loss_factor = 0.5
        self.landmark_loss_factor = 0.5

        self.factory = PNet

        # config for database to train net
        self.dbase = None

        # prefix to save trained net
        self.prefix = None


# construct PNet
class PNet:
    def __init__(self, inputs, label=None, bbox_target=None, landmark_target=None, training=True):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='valid'):
            print(inputs.get_shape())
            net = slim.conv2d(inputs, 10, 3, stride=1, scope='conv1')
            activation_summary(net)
            print(net.get_shape())
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1', padding='SAME')
            activation_summary(net)
            print(net.get_shape())
            net = slim.conv2d(net, num_outputs=16, kernel_size=[3, 3], stride=1, scope='conv2')
            activation_summary(net)
            print(net.get_shape())
            net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=1, scope='conv3')
            activation_summary(net)
            print(net.get_shape())
            conv4_1 = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1], stride=1, scope='conv4_1', activation_fn=tf.nn.softmax)
            activation_summary(conv4_1)
            print(conv4_1.get_shape())
            bbox_pred = slim.conv2d(net, num_outputs=4, kernel_size=[1, 1], stride=1, scope='conv4_2', activation_fn=None)
            activation_summary(bbox_pred)
            print(bbox_pred.get_shape())
            landmark_pred = slim.conv2d(net, num_outputs=10, kernel_size=[1, 1], stride=1, scope='conv4_3', activation_fn=None)
            activation_summary(landmark_pred)
            print(landmark_pred.get_shape())

            if training:
                cls_prob = tf.squeeze(conv4_1, [1, 2], name='cls_prob')
                self.cls_loss = cls_ohem(cls_prob, label)
                bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
                self.bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                landmark_pred = tf.squeeze(landmark_pred, [1, 2], name='landmark_pred')
                self.landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
                self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())

                tp, tn, fp, fn = contingency_table(cls_prob, label)

                self.accuracy = tf.divide(tp + tn, tp + tn + fp + fn, name='accuracy')
                self.precision = tf.divide(tp, tp + fp, name='precision')
                self.recall = tf.divide(tp, tp + fn, name='recall')
            else:
                self.cls_pro_test = tf.squeeze(conv4_1, axis=0)
                print(self.cls_pro_test)
                self.bbox_pred_test = tf.squeeze(bbox_pred, axis=0)
                print(self.bbox_pred_test)
                self.landmark_pred_test = tf.squeeze(landmark_pred, axis=0)
                print(self.landmark_pred_test)

    def loss(self, config):
        loss = config.cls_loss_factor * self.cls_loss + \
               config.bbox_loss_factor * self.bbox_loss + \
               config.landmark_loss_factor * self.landmark_loss + \
               self.l2_loss

        return loss
