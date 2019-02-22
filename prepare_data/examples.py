# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import cv2

from prepare_data import wider
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from prepare_data import ioutils, h5utils
from prepare_data.utils import convert_to_square
from prepare_data.data_utils import IoU


def generate(dbwider, models, threshold=(0.6, 0.7, 0.7), min_face_size=25, stride=2, slide_window=False):

    detectors = [None, None, None]

    if len(models) < 2:
        raise ValueError('the number of models must be 2 or 3.')

    batch_size = 256

    # load P-Net model (detectors[0]) to generate data to train R-Net or O-Net
    config = models[0]
    model_path = '{}-{}'.format(config.prefix, config.number_of_epochs)

    if slide_window:
        detectors[0] = Detector(config.factory, config.image_size, batch_size, model_path)
    else:
        detectors[0] = FcnDetector(config.factory, model_path)

    image_size = models[1].image_size
    h5file = models[1].dbase.h5file

    # load R-Net model (detectors[1]) to generate data to train O-Net
    if len(models) > 2:
        config = models[1]
        model_path = '{}-{}'.format(config.prefix, config.number_of_epochs)
        detectors[1] = Detector(config.factory, config.image_size, batch_size, model_path)

        image_size = models[2].image_size
        h5file = models[2].dbase.h5file

    # initialize detector
    detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                             stride=stride, threshold=threshold, slide_window=slide_window)

    # create output directories
    for key in ('positive', 'negative', 'part'):
        outdir = h5file.parent.joinpath(key)
        if not outdir.exists():
            outdir.mkdir(parents=True)

    data = ioutils.read_annotation(dbwider.wider_face_train_bbx_gt)
    # data['images'] = data['images'][:500]
    # data['bboxes'] = data['bboxes'][:500]

    # index of negative, positive and part face, used as their image names
    positive = []
    negative = []
    part = []

    loader = ioutils.ImageLoader(data['images'], prefix=dbwider.images)

    for img, gts in zip(loader, data['bboxes']):
        dets, _ = detector.detect_single_image(img)
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
                key_name = os.path.join('negative', '{}.jpg'.format(len(negative)))
                ioutils.write_image(resized, key_name, prefix=h5file.parent)
                negative.append((key_name, 0, 0, 0, 0, 0))
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
                    key_name = os.path.join('positive', '{}.jpg'.format(len(positive)))
                    ioutils.write_image(resized, key_name, prefix=h5file.parent)
                    positive.append((key_name, 1, offset_x1, offset_y1, offset_x2, offset_y2))

                elif np.max(iou_values) >= 0.4:
                    key_name = os.path.join('part', '{}.jpg'.format(len(part)))
                    ioutils.write_image(resized, key_name, prefix=h5file.parent)
                    part.append((key_name, -1, offset_x1, offset_y1, offset_x2, offset_y2))

    h5utils.write(h5file, 'positive', np.array(positive, dtype=wider.dtype))
    h5utils.write(h5file, 'negative', np.array(negative, dtype=wider.dtype))
    h5utils.write(h5file, 'part', np.array(part, dtype=wider.dtype))
