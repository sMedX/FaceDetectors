# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import cv2
import pathlib as plib
import tensorflow as tf
import contextlib

import tfmtcnn
from tfmtcnn.prepare_data import tfrecords
from tfmtcnn.prepare_data import ioutils
from tfmtcnn.prepare_data.utils import IoU
from tfmtcnn.models import pnet, rnet, onet
from tfmtcnn.mtcnn import MTCNN
from tfmtcnn.prepare_data.utils import convert_to_square

"""
http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
WIDER FACE dataset is a face detection benchmark dataset, of which images are selected from the publicly available WIDER 
dataset. We choose 32,203 images and label 393,703 faces with a high degree of variability in scale, pose and occlusion 
as depicted in the sample images. WIDER FACE dataset is organized based on 61 event classes. For each event class, 
we randomly select 40%/10%/50% data as training, validation and testing sets. We adopt the same evaluation metric 
employed in the PASCAL VOC dataset. Similar to MALF and Caltech datasets, we do not release bounding box ground truth 
for the test images. Users are required to submit final prediction files, which we shall proceed to evaluate.
"""


class DBWider:
    def __init__(self, path):
        self.labels = ('positive', 'negative', 'part')

        self.path = plib.Path(os.path.expanduser(path)).absolute()
        self.images = self.path.joinpath('images')
        self.wider_face_train = tfmtcnn.dirname().joinpath('data', 'wider_face_train.txt')
        self.wider_face_train_bbx_gt = tfmtcnn.dirname().joinpath('data', 'wider_face_train_bbx_gt.txt')
        self.tfwriter = None

    def prepare(self, tfprefix, image_size, seed=None):
        np.random.seed(seed=seed)

        ioutils.mkdir(tfprefix.parent)

        def writer(label):
            tffile = tfrecords.getfilename(tfprefix, label)
            return tf.python_io.TFRecordWriter(str(tffile))

        with contextlib.ExitStack() as stack:
            self.tfwriter = {label: stack.enter_context(writer(label)) for label in self.labels}
            self._prepare(image_size)

    def _prepare(self, image_size):

        with self.wider_face_train.open() as f:
            annotations = [a.strip() for a in f]
        number_of_images = len(annotations)

        print('number of images {}'.format(number_of_images))

        files = []
        list_of_boxes = []
        for annotation in annotations:
            annotation = annotation.split(' ')
            files.append(annotation[0] + '.jpg')
            bbox = list(map(float, annotation[1:]))
            list_of_boxes.append(np.array(bbox, dtype=np.float32).reshape(-1, 4))

        loader = ioutils.ImageLoader(files, prefix=self.images)

        for img, boxes in zip(loader, list_of_boxes):
            height, width, channel = img.shape

            # keep crop random parts, until have 50 negative examples get 50 negative sample from every image
            for i in range(50):
                # neg_num's size [40,min(width, height) / 2], min_size:40
                # size is a random number between 12 and min(width,height)
                size = np.random.randint(image_size, min(width, height) / 2)

                # top_left coordinate
                nx = np.random.randint(0, width - size)
                ny = np.random.randint(0, height - size)

                # random crop
                crop_box = np.array([nx, ny, nx + size, ny + size])

                # calculate iou
                iou_values = IoU(crop_box, boxes)

                if np.max(iou_values) < 0.3:
                    # resize the cropped image to size 12*12
                    cropped = img[ny: ny + size, nx: nx + size, :]
                    resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                    # Iou with all gts must below 0.3
                    self.add_to_tfrecord('negative', resized, [0, 0, 0, 0, 0])

            # for every bounding boxes
            for box in boxes:
                # box (x_left, y_top, x_right, y_bottom)
                x1, y1, x2, y2 = box
                # gt's width
                w = x2 - x1 + 1
                # gt's height
                h = y2 - y1 + 1

                # ignore small faces and those faces has left-top corner out of the image
                # in case the ground truth boxes of small faces are not accurate
                if max(w, h) < 20 or x1 < 0 or y1 < 0:
                    continue

                # crop another 5 images near the bounding box if IoU less than 0.5, save as negative samples
                for i in range(5):
                    # size of the image to be cropped
                    size = np.random.randint(image_size, min(width, height) / 2)
                    # delta_x and delta_y are offsets of (x1, y1)
                    # max can make sure if the delta is a negative number , x1+delta_x >0
                    # parameter high of randint make sure there will be intersection between bbox and cropped_box
                    delta_x = np.random.randint(max(-size, -x1), w)
                    delta_y = np.random.randint(max(-size, -y1), h)
                    # max here not really necessary
                    nx1 = int(max(0, x1 + delta_x))
                    ny1 = int(max(0, y1 + delta_y))
                    # if the right bottom point is out of image then skip
                    if nx1 + size > width or ny1 + size > height:
                        continue
                    crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                    iou_values = IoU(crop_box, boxes)

                    if np.max(iou_values) < 0.3:
                        # resize cropped image to be 12 * 12
                        cropped = img[ny1: ny1 + size, nx1: nx1 + size, :]
                        resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                        # Iou with all gts must below 0.3
                        self.add_to_tfrecord('negative', resized, [0, 0, 0, 0, 0])

                # generate positive examples and part faces
                for i in range(20):
                    # pos and part face size [minsize*0.8,maxsize*1.25]
                    size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                    # delta here is the offset of box center
                    if w < 5:
                        continue
                    # print (box)
                    delta_x = np.random.randint(-0.2 * w, 0.2 * w)
                    delta_y = np.random.randint(-0.2 * h, 0.2 * h)

                    # show this way: nx1 = max(x1+w/2-size/2+delta_x)
                    # x1+ w/2 is the central point, then add offset , then deduct size/2
                    # deduct size/2 to make sure that the right bottom corner will be out of
                    nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                    # show this way: ny1 = max(y1+h/2-size/2+delta_y)
                    ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                    nx2 = nx1 + size
                    ny2 = ny1 + size

                    if nx2 > width or ny2 > height:
                        continue
                    crop_box = np.array([nx1, ny1, nx2, ny2])

                    # offset
                    offset_x1 = (x1 - nx1) / float(size)
                    offset_y1 = (y1 - ny1) / float(size)
                    offset_x2 = (x2 - nx2) / float(size)
                    offset_y2 = (y2 - ny2) / float(size)

                    # crop and resize image
                    cropped = img[ny1: ny2, nx1: nx2, :]
                    resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                    box_ = box.reshape(1, -1)
                    iou = IoU(crop_box, box_)
                    if iou >= 0.65:
                        sample = [1, offset_x1, offset_y1, offset_x2, offset_y2]
                        self.add_to_tfrecord('positive', resized, sample)

                    elif iou >= 0.4:
                        sample = [-1, offset_x1, offset_y1, offset_x2, offset_y2]
                        self.add_to_tfrecord('part', resized, sample)

    def prepare_with_mtcnn(self, configs, threshold=(0.6, 0.7, 0.7), min_face_size=20, stride=2):

        if not len(configs) in (2, 3):
            raise ValueError('the number of configs must be 2 or 3.')

        tfprefix = configs[len(configs)-1].dbase.tfprefix
        ioutils.mkdir(tfprefix.parent)

        def writer(label):
            tffile = tfrecords.getfilename(tfprefix, label)
            return tf.python_io.TFRecordWriter(str(tffile))

        with contextlib.ExitStack() as stack:
            self.tfwriter = {label: stack.enter_context(writer(label)) for label in self.labels}
            self._hardexamples(configs, threshold, min_face_size, stride)

    def _hardexamples(self, configs, threshold, min_face_size, stride):

        detectors = [None, None, None]

        batch_size = 256

        for i, (config, net) in enumerate(zip(configs, (pnet.PNet, rnet.RNet, onet.ONet))):
            if i < len(configs)-1:
                model_path = '{}-{}'.format(config.prefix, config.number_of_epochs)
                detectors[i] = net(config, batch_size, model_path)
            else:
                image_size = config.image_size

        # load P-Net model (detectors[0]) to generate data to train R-Net or O-Net
        # config = configs[0]
        # model_path = '{}-{}'.format(config.prefix, config.number_of_epochs)
        # detectors[0] = pnet.PNet(config, model_path)
        # image_size = configs[1].image_size
        # if len(configs) > 2:
        #     # load R-Net model (detectors[1]) to generate data to train O-Net
        #     config = configs[1]
        #     model_path = '{}-{}'.format(config.prefix, config.number_of_epochs)
        #     detectors[1] = rnet.RNet(config, batch_size, model_path)
        #     image_size = configs[2].image_size

        # initialize detector
        detector = MTCNN(detectors=detectors,
                         min_face_size=min_face_size,
                         stride=stride,
                         threshold=threshold)

        data = ioutils.read_annotation(self.images, self.wider_face_train_bbx_gt)
        # data['images'] = data['images'][:500]
        # data['bboxes'] = data['bboxes'][:500]

        # index of negative, positive and part face, used as their image names
        # positive = []
        # negative = []
        # part = []

        loader = ioutils.ImageLoader(data['images'], prefix=self.images)

        for img, gts in zip(loader, data['bboxes']):
            dets, _ = detector.detect(img)
            if dets.shape[0] == 0:
                continue

            # change to square
            dets = convert_to_square(dets)
            dets[:, 0:4] = np.round(dets[:, 0:4])

            gts = np.array(gts, dtype=np.float32).reshape(-1, 4)

            neg_num = 0

            for box in dets:
                x_left, y_top, x_right, y_bottom, _ = box.astype(int)
                width = x_right - x_left + 1
                height = y_bottom - y_top + 1

                # ignore box that is too small or beyond image border
                if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                    continue

                # compute intersection over union(IoU) between current box and all gt boxes
                iou_values = IoU(box, gts)

                cropped = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
                resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                # save negative images and write label Iou with all gts must below 0.3
                if np.max(iou_values) < 0.3 and neg_num < 60:
                    # key_name = os.path.join('negative', '{}.jpg'.format(len(negative)))
                    # ioutils.write_image(resized, key_name, prefix=h5file.parent)
                    # negative.append((key_name, 0, 0, 0, 0, 0))
                    self.add_to_tfrecord('negative', resized, [0, 0, 0, 0, 0])
                    neg_num += 1
                else:
                    # find gt_box with the highest iou
                    idx = np.argmax(iou_values)
                    assigned_gt = gts[idx]
                    x1, y1, x2, y2 = assigned_gt

                    # compute bbox reg label
                    offset_x1 = (x1 - x_left) / float(width)
                    offset_y1 = (y1 - y_top) / float(height)
                    offset_x2 = (x2 - x_right) / float(width)
                    offset_y2 = (y2 - y_bottom) / float(height)

                    # save positive and part-face images and write labels
                    if np.max(iou_values) >= 0.65:
                        # key_name = os.path.join('positive', '{}.jpg'.format(len(positive)))
                        # ioutils.write_image(resized, key_name, prefix=h5file.parent)
                        # positive.append((key_name, 1, offset_x1, offset_y1, offset_x2, offset_y2))
                        self.add_to_tfrecord('positive', resized, [1, offset_x1, offset_y1, offset_x2, offset_y2])

                    elif np.max(iou_values) >= 0.4:
                        # key_name = os.path.join('part', '{}.jpg'.format(len(part)))
                        # ioutils.write_image(resized, key_name, prefix=h5file.parent)
                        # part.append((key_name, -1, offset_x1, offset_y1, offset_x2, offset_y2))
                        self.add_to_tfrecord('part', resized, [-1, offset_x1, offset_y1, offset_x2, offset_y2])

    def add_to_tfrecord(self, label, image, sample):
        image_buffer = image.tostring()

        class_label = sample[0]
        roi = sample[1:1+4]
        landmark = [0]*10

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': tfrecords.bytes_feature(image_buffer),
            'image/label': tfrecords.int64_feature(class_label),
            'image/roi': tfrecords.float_feature(roi),
            'image/landmark': tfrecords.float_feature(landmark)
        }))

        self.tfwriter[label].write(example.SerializeToString())
