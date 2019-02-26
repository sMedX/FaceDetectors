# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import pathlib as plib
from tensorflow.contrib import slim
from .mtcnn import *


# config to train R-Net (prediction net)
class Factory:
    def __init__(self, model_path=None):
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

        self.factory = RNet

        # config for database to train net
        self.dbase = None

        # prefix to save trained net
        self.prefix = None

        self.model_path = model_path

    @property
    def detector(self):
        detector = Detector(self, model_path=self.model_path)
        return detector


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
                self.landmark_loss = landmark_ohem(self.landmark_pred, landmark_target, label)
                self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())

                tp, tn, fp, fn = contingency_table(self.cls_prob, label)

                self.accuracy = tf.divide(tp + tn, tp + tn + fp + fn, name='accuracy')
                self.precision = tf.divide(tp, tp + fp, name='precision')
                self.recall = tf.divide(tp, tp + fn, name='recall')

    def loss(self, config):
        loss = config.cls_loss_factor * self.cls_loss + \
               config.bbox_loss_factor * self.bbox_loss + \
               config.landmark_loss_factor * self.landmark_loss + \
               self.l2_loss

        return loss


class Detector:
    def __init__(self, config, batch_size=1, model_path=None):
        size = config.image_size

        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, size, size, 3], name='input_image')

            net = RNet(self.image_op, training=False)
            self.cls_prob = net.cls_prob
            self.bbox_pred = net.bbox_pred
            self.landmark_pred = net.landmark_pred
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                         gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()

            if model_path is None:
                model_path = plib.Path(__file__).parent.joinpath('parameters', 'rnet', 'rnet')

            try:
                saver.restore(self.sess, str(model_path))
            except:
                raise IOError('unable restore parameters from {}'.format(str(model_path)))

            print('restore parameters from the path {}'.format(str(model_path)))

        self.data_size = size
        self.batch_size = batch_size

    def predict(self, databatch):
        # databatch: N x 3 x data_size x data_size
        batch_size = self.batch_size

        minibatch = []
        cur = 0
        n = databatch.shape[0]
        while cur < n:
            #split mini-batch
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        #every batch prediction result
        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            #the last batch
            if m < batch_size:
                keep_inds = np.arange(m)
                #gap (difference)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            #cls_prob batch*2
            #bbox_pred batch*4
            cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.image_op: data})
            #num_batch * batch_size *2
            cls_prob_list.append(cls_prob[:real_size])
            #num_batch * batch_size *4
            bbox_pred_list.append(bbox_pred[:real_size])
            #num_batch * batch_size*10
            landmark_pred_list.append(landmark_pred[:real_size])
            #num_of_data*2,num_of_data*4,num_of_data*10
        return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)
