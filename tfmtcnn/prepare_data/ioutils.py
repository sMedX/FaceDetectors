# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import cv2
import time
import pathlib as plib


def write_image(image, filename, prefix=None):
    if prefix is not None:
        filename = os.path.join(str(prefix), str(filename))

    if not cv2.imwrite(str(filename), image):
        raise IOError('while writing the file {}'.format(filename))


def read_image(filename, prefix=None):
    if prefix is not None:
        image = cv2.imread(os.path.join(str(prefix), str(filename)))
    else:
        image = cv2.imread(str(filename))

    if image is None:
        raise IOError('while reading the file {}'.format(filename))
    return image


class ImageLoader:
    def __iter__(self):
        return self

    def __init__(self, data, prefix=None, display=100):
        self.counter = -1
        self.start_time = time.time()
        self.data = data
        self.display = display
        self.size = len(data)
        self.prefix = str(prefix)
        self.path = None

    def __next__(self):
        self.counter += 1

        if self.counter < self.size:
            if self.counter % self.display == 0:
                elapsed_time = (time.time() - self.start_time) / self.display
                print('\rnumber of processed images {}/{}, {:.5f} seconds per image'.
                      format(self.counter, self.size, elapsed_time), end='')
                self.start_time = time.time()

            self.path = plib.Path(os.path.join(str(self.prefix), self.data[self.counter]))
            image = read_image(self.path)

            return image
        else:
            print('\rnumber of processed images {}'.format(self.size))
            raise StopIteration

    def reset(self):
        self.counter = -1
        return self


def read_annotation(filename):

    if not filename.exists():
        raise IOError('file {} does not exist'.format(filename))
    data = dict()
    images = []
    bboxes = []
    labelfile = open(str(filename), 'r')

    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        # imagepath = base_dir + '/WIDER_train/images/' + imagepath
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        # im = cv2.imread(imagepath)
        # h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums)):
            # text = ''
            # text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            # text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            # text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            # f.write(text + '\n')
        bboxes.append(one_image_bboxes)

    data['images'] = images
    data['bboxes'] = bboxes

    return data
