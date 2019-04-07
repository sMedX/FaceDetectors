# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import tfmtcnn
from collections import OrderedDict
from tensorflow.contrib import slim
from tfmtcnn.models.mtcnn import *


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

        self.cls_loss_factor = 1.0
        self.bbox_loss_factor = 0.5
        self.landmark_loss_factor = 0.5

        # config for database to train net
        self.dbase = None

        # prefix to save trained net
        self.prefix = None


# construct RNet
class RNet:
    def __init__(self, config=None, batch_size=1, model_path=None):
        if config is None:
            config = Config()
        self.config = config

        if model_path == 'default':
            model_path = tfmtcnn.dirname().joinpath('models', 'parameters', 'rnet', 'rnet')
        self.model_path = model_path

        if model_path is not None:
            graph = tf.Graph()
            size = config.image_size
            self.batch_size = batch_size

            with graph.as_default():
                self.image_op = tf.placeholder(tf.float32, shape=[batch_size, size, size, 3], name='input_image')

                self.cls_prob, self.bbox_pred, self.landmark_pred = self.activate(self.image_op)
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

            cls_prob = slim.fully_connected(fc1, num_outputs=2, scope='cls_fc', activation_fn=tf.nn.softmax)
            print(cls_prob.get_shape())

            bbox_pred = slim.fully_connected(fc1, num_outputs=4, scope='bbox_fc', activation_fn=None)
            print(bbox_pred.get_shape())

            landmark_pred = slim.fully_connected(fc1, num_outputs=10, scope='landmark_fc', activation_fn=None)
            print(landmark_pred.get_shape())

        return cls_prob, bbox_pred, landmark_pred

    def loss(self, inputs, label, bbox_target, landmark_target):
        cls_prob, bbox_pred, landmark_pred = self.activate(inputs)

        cls_loss = cls_ohem(cls_prob, label)
        bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
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

    def predict(self, databatch):
        batch_size = self.batch_size

        minibatch = []

        cur = 0
        n = databatch.shape[0]

        while cur < n:
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size

        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []

        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size

            if m < batch_size:
                keep_inds = np.arange(m)
                gap = self.batch_size - m

                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))

                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))

                data = data[keep_inds]
                real_size = m

            cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.image_op: data})
            cls_prob_list.append(cls_prob[:real_size])
            bbox_pred_list.append(bbox_pred[:real_size])
            landmark_pred_list.append(landmark_pred[:real_size])

        return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)
