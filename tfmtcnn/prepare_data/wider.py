# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import h5py
import numpy as np
from tfmtcnn.prepare_data import ioutils, h5utils
from tfmtcnn.prepare_data.utils import IoU
import cv2
import pathlib as plib

dtype = np.dtype([('path', h5py.special_dtype(vlen=str)),
                  ('label', np.int8),
                  ('1', np.float),
                  ('2', np.float),
                  ('3', np.float),
                  ('4', np.float)])


class DBWider:
    def __init__(self, path):
        self.path = plib.Path(os.path.expanduser(path)).absolute()
        self.images = self.path.joinpath('images')
        self.wider_face_train = self.path.joinpath('wider_face_train.txt')
        self.wider_face_train_bbx_gt = self.path.joinpath('wider_face_train_bbx_gt.txt')

    def read_wider_face_train_bbx_gt(self):
        pass


labels = ('positive', 'negative', 'part')


def prepare(dbase, outdir, image_size=12, seed=None):
    np.random.seed(seed=seed)

    positive = []
    negative = []
    part = []

    outdir.positive = outdir.output.joinpath('positive')
    outdir.negative = outdir.output.joinpath('negative')
    outdir.part = outdir.output.joinpath('part')

    for dir in (outdir.positive, outdir.negative, outdir.part):
        if not dir.exists():
            dir.mkdir(parents=True)

    with dbase.wider_face_train.open() as f:
        annotations = [a.strip() for a in f]
    number_of_images = len(annotations)

    print('number of images {}'.format(number_of_images))
    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # don't care

    files = []
    list_of_boxes = []
    for annotation in annotations:
        annotation = annotation.split(' ')
        files.append(annotation[0] + '.jpg')
        bbox = list(map(float, annotation[1:]))
        list_of_boxes.append(np.array(bbox, dtype=np.float32).reshape(-1, 4))

    loader = ioutils.ImageLoader(files, prefix=dbase.images)

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
                filename = outdir.negative.joinpath('{}.jpg'.format(n_idx))
                ioutils.write_image(resized, filename)
                negative.append((os.path.join(filename.parent.name, filename.name), 0, 0, 0, 0, 0))
                n_idx += 1

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
                    filename = outdir.negative.joinpath('{}.jpg'.format(n_idx))
                    ioutils.write_image(resized, filename)
                    negative.append((os.path.join(filename.parent.name, filename.name), 0, 0, 0, 0, 0))
                    n_idx += 1

            # generate positive examples and part faces
            for i in range(20):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                if w < 5:
                    continue
                # print (box)
                delta_x = np.random.randint(-0.2*w, 0.2*w)
                delta_y = np.random.randint(-0.2*h, 0.2*h)

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
                    filename = outdir.positive.joinpath('{}.jpg'.format(p_idx))
                    ioutils.write_image(resized, filename)
                    positive.append((os.path.join(filename.parent.name, filename.name), 1, offset_x1, offset_y1, offset_x2, offset_y2))

                    p_idx += 1

                elif iou >= 0.4:
                    filename = outdir.part.joinpath('{}.jpg'.format(d_idx))
                    ioutils.write_image(resized, filename)
                    part.append((os.path.join(filename.parent.name, filename.name), -1, offset_x1, offset_y1, offset_x2, offset_y2))

                    d_idx += 1

    print('\r{} images have been processed, positive: {}, negative: {}, part: {}'.
          format(number_of_images, p_idx, n_idx, d_idx))

    h5utils.write(outdir.h5file, 'positive', np.array(positive, dtype=dtype))
    h5utils.write(outdir.h5file, 'negative', np.array(negative, dtype=dtype))
    h5utils.write(outdir.h5file, 'part', np.array(negative, dtype=dtype))