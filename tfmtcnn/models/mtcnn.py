# coding:utf-8

import numpy as np
import tensorflow as tf

num_keep_radio = 0.7


# define prelu
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    # get the number of rows of class_prob
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    # set pos and neg to be 1, rest to be 0
    valid_inds = tf.where(label < zeros,zeros,ones)
    # get the number of POS and NEG examples
    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def bbox_ohem_smooth_l1_loss(bbox_pred,bbox_target,label):
    sigma = tf.constant(1.0)
    threshold = 1.0/(sigma**2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    abs_error = tf.abs(bbox_pred-bbox_target)
    loss_smaller = 0.5*((abs_error*sigma)**2)
    loss_larger = abs_error-0.5/(sigma**2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error<threshold,loss_smaller,loss_larger),axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    smooth_loss = smooth_loss*valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)


def bbox_ohem_orginal(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    #pay attention :there is a bug!!!!
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    #(batch,)
    square_error = tf.reduce_sum(tf.square(bbox_pred-bbox_target),axis=1)
    #keep_num scalar
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)


# label=1 or label=-1 then do regression
def bbox_ohem(bbox_pred, bbox_target, label):
    '''

    :param bbox_pred:
    :param bbox_target:
    :param label: class label
    :return: mean euclidean loss for all the pos and part examples
    '''
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)
    # keep pos and part examples
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    # (batch,)
    # calculate square sum
    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    # keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    # keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    # count the number of pos and part examples
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # keep valid index square_error
    square_error = square_error*valid_inds
    # keep top k examples, k equals to the number of positive examples
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)


def landmark_ohem(landmark_pred, landmark_target, label):
    """

    :param landmark_pred:
    :param landmark_target:
    :param label:
    :return:
    """
    ones = tf.ones_like(label, dtype=tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)


def accuracy(cls_prob, label):
    """
    :param cls_prob:
    :param label:
    :return:calculate classification accuracy for pos and neg examples only
    """
    # get the index of maximum value along axis one from cls_prob 0 for negative 1 for positive
    pred = tf.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int64)

    # return the index of pos and neg examples
    cond = tf.where(tf.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)

    # gather the label of pos and neg examples
    label_picked = tf.gather(label_int, picked)
    pred_picked = tf.gather(pred, picked)

    # calculate the mean value of a vector contains 1 and 0, 1 for correct classification, 0 for incorrect
    # ACC = (TP+FP)/total population
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked, pred_picked), tf.float32))
    return accuracy_op


def activation_summary(x):
    tensor_name = x.op.name
    print('load summary for : ', tensor_name)
    tf.summary.histogram(tensor_name + '/activations', x)
