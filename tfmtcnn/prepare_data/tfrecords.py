# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import tensorflow as tf
from tfmtcnn.prepare_data import h5utils, ioutils


def getfilename(prefix, key):
    return prefix.with_name(prefix.name + key).with_suffix('.tfrecord')


def int64_feature(value):
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def add_to_tfrecord(writer, filename, data):
    image = ioutils.read_image(filename)
    image_buffer = image.tostring()

    class_label = data['label']
    bbox = data['bbox']
    roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    landmark = [bbox['xlefteye'], bbox['ylefteye'], bbox['xrighteye'], bbox['yrighteye'], bbox['xnose'], bbox['ynose'],
                bbox['xleftmouth'], bbox['yleftmouth'], bbox['xrightmouth'], bbox['yrightmouth']]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_buffer),
        'image/label': int64_feature(class_label),
        'image/roi': float_feature(roi),
        'image/landmark': float_feature(landmark)
    }))

    writer.write(example.SerializeToString())


def data2sample(inpdata):

    outdata = []
    rect = ('xmin', 'ymin', 'xmax', 'ymax')
    landmarks = ('xlefteye', 'ylefteye',
                 'xrighteye', 'yrighteye',
                 'xnose', 'ynose',
                 'xleftmouth', 'yleftmouth',
                 'xrightmouth', 'yrightmouth')

    for values in inpdata:
        sample = dict()
        sample['filename'] = values[0]
        sample['label'] = values[1]
        sample['bbox'] = {x: 0 for x in rect + landmarks}

        values = list(values)[2:]

        if len(values) == 4:
            for key, value in zip(rect, values):
                sample['bbox'][key] = value
        else:
            for key, value in zip(landmarks, values):
                sample['bbox'][key] = value
        outdata.append(sample)

    return outdata


def write_single_tfrecord(h5file, tffile, key=None, size=None, seed=None):
    """

    :param h5file:
    :param tffile:
    :param key:
    :param size:
    :param seed:
    :return:
    """
    np.random.seed(seed=seed)

    # tf record file name
    if tffile.exists():
        os.remove(str(tffile))

    # get data from the h5 file
    data = h5utils.read(h5file, key)

    if size is None:
        size = len(data)
    if size < len(data):
        data = np.random.choice(data, size=size)

    tfdata = data2sample(data)

    np.random.shuffle(tfdata)

    with tf.python_io.TFRecordWriter(str(tffile)) as writer:
        for i, sample in enumerate(tfdata):
            filename = h5file.parent.joinpath(sample['filename'])
            add_to_tfrecord(writer, filename, sample)

            if (i+1) % 100 == 0:
                print('\r{}/{} samples have been added to tfrecord file.'.format(i+1, len(tfdata)), end='')

    print('\rtfrecord file {} has been written, number of samples is {}.'.format(tffile, len(tfdata)))


def write_multi_tfrecords(h5file, prefix=None, seed=None):

    files = []

    for key in h5utils.keys(h5file):
        filename = getfilename(prefix, key)
        write_single_tfrecord(h5file, filename, key=key, seed=seed)
        files.append(filename)

    return files


def read_single_tfrecord(config, tfrecord, batch_size):
    # generate an input queue each epoch shuffle
    filename_queue = tf.train.string_input_producer([str(tfrecord)], shuffle=True)

    # read tfrecord
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/label': tf.FixedLenFeature([], tf.int64),
            'image/roi': tf.FixedLenFeature([4], tf.float32),
            'image/landmark': tf.FixedLenFeature([10], tf.float32)
        }
    )

    image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [config.image_size, config.image_size, 3])
    image = (tf.cast(image, tf.float32) - 127.5) / 128

    # image = tf.image.per_image_standardization(image)
    label = tf.cast(image_features['image/label'], tf.float32)
    roi = tf.cast(image_features['image/roi'], tf.float32)
    landmark = tf.cast(image_features['image/landmark'], tf.float32)
    image, label, roi, landmark = tf.train.batch([image, label, roi, landmark],
                                                 batch_size=batch_size, num_threads=2, capacity=batch_size)
    label = tf.reshape(label, [batch_size])
    roi = tf.reshape(roi, [batch_size, 4])
    landmark = tf.reshape(landmark, [batch_size, 10])
    return image, label, roi, landmark


def read_multi_tfrecords(config, tfrecords, batch_sizes):
    """

    :param config:
    :param tfrecords:
    :param batch_sizes:
    :return:
    """
    pos_dir, part_dir, neg_dir, landmark_dir = tfrecords
    pos_batch_size, part_batch_size, neg_batch_size, landmark_batch_size = batch_sizes

    pos_image, pos_label, pos_roi, pos_landmark = read_single_tfrecord(config, pos_dir, pos_batch_size)
    print(pos_image.get_shape())

    part_image, part_label, part_roi, part_landmark = read_single_tfrecord(config, part_dir, part_batch_size)
    print(part_image.get_shape())

    neg_image, neg_label, neg_roi, neg_landmark = read_single_tfrecord(config, neg_dir, neg_batch_size)
    print(neg_image.get_shape())

    landmark_image, landmark_label, landmark_roi, landmark_landmark = read_single_tfrecord(config, landmark_dir,
                                                                                           landmark_batch_size)
    print(landmark_image.get_shape())

    images = tf.concat([pos_image, part_image, neg_image, landmark_image], 0, name="concat/image")
    print(images.get_shape())
    labels = tf.concat([pos_label, part_label, neg_label, landmark_label], 0, name="concat/label")

    assert isinstance(labels, object)
    labels.get_shape()
    rois = tf.concat([pos_roi, part_roi, neg_roi, landmark_roi], 0, name="concat/roi")
    print(rois.get_shape())
    landmarks = tf.concat([pos_landmark, part_landmark, neg_landmark, landmark_landmark], 0, name="concat/landmark")
    return images, labels, rois, landmarks
