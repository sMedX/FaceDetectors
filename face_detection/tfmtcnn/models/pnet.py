# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

from collections import OrderedDict
from tensorflow.contrib import slim

from face_detection import tfmtcnn
from face_detection.tfmtcnn.models.mtcnn import *


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

        # config for database to train net
        self.dbase = None

        # prefix to save trained net
        self.prefix = None


# construct PNet
class PNet:
    def __init__(self, config=None, batch_size=1, model_path=None):
        if config is None:
            config = Config()
        self.config = config

        if model_path == 'default':
            model_path = tfmtcnn.dirname().joinpath('models', 'parameters', 'pnet', 'pnet')
        self.model_path = model_path

        # create a graph
        if model_path is not None:
            graph = tf.Graph()
            with graph.as_default():
                self.image_op = tf.placeholder(tf.float32, name='input_image')
                self.width_op = tf.placeholder(tf.int32, name='image_width')
                self.height_op = tf.placeholder(tf.int32, name='image_height')
                image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])

                cls_prob, bbox_pred, landmark_pred = self.activate(image_reshape)

                self.cls_prob = tf.squeeze(cls_prob, axis=0)
                print(self.cls_prob)

                self.bbox_pred = tf.squeeze(bbox_pred, axis=0)
                print(self.bbox_pred)

                self.landmark_pred = tf.squeeze(landmark_pred, axis=0)
                print(self.landmark_pred)

                self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                             gpu_options=tf.GPUOptions(allow_growth=True)))

                try:
                    tf.train.Saver().restore(self.sess, str(self.model_path))
                except:
                    raise IOError('unable restore parameters from {}'.format(str(self.model_path)))

                print('restore parameters from the path {}'.format(str(self.model_path)))

    @staticmethod
    def activate(inputs):
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

            cls_prob = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1], stride=1, scope='conv4_1', activation_fn=tf.nn.softmax)
            activation_summary(cls_prob)
            print(cls_prob.get_shape())

            bbox_pred = slim.conv2d(net, num_outputs=4, kernel_size=[1, 1], stride=1, scope='conv4_2', activation_fn=None)
            activation_summary(bbox_pred)
            print(bbox_pred.get_shape())

            landmark_pred = slim.conv2d(net, num_outputs=10, kernel_size=[1, 1], stride=1, scope='conv4_3', activation_fn=None)
            activation_summary(landmark_pred)
            print(landmark_pred.get_shape())

            return cls_prob, bbox_pred, landmark_pred

    def loss(self, inputs, label=None, bbox_target=None, landmark_target=None):
        cls_prob, bbox_pred, landmark_pred = self.activate(inputs)

        cls_prob = tf.squeeze(cls_prob, [1, 2], name='cls_prob')
        cls_loss = cls_ohem(cls_prob, label)
        bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
        bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
        landmark_pred = tf.squeeze(landmark_pred, [1, 2], name='landmark_pred')
        landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
        l2_loss = tf.add_n(slim.losses.get_regularization_losses())

        tp, tn, fp, fn = contingency_table(cls_prob, label)

        accuracy = tf.divide(tp + tn, tp + tn + fp + fn, name='accuracy')
        precision = tf.divide(tp, tp + fp, name='precision')
        recall = tf.divide(tp, tp + fn, name='recall')

        total_loss = self.config.cls_loss_factor * cls_loss + self.config.bbox_loss_factor * bbox_loss + self.config.landmark_loss_factor * landmark_loss + l2_loss

        metrics = OrderedDict()
        metrics['total_loss'] = total_loss
        metrics['class_loss'] = cls_loss
        metrics['bbox_loss'] = bbox_loss
        metrics['landmark_loss'] = landmark_loss
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['accuracy'] = accuracy

        return total_loss, metrics

    def predict(self, batch):
        height, width, channels = batch.shape
        cls_prob, bbox = self.sess.run([self.cls_prob, self.bbox_pred],
                                       feed_dict={self.image_op: batch,
                                                  self.width_op: width,
                                                  self.height_op: height})
        return cls_prob, bbox
