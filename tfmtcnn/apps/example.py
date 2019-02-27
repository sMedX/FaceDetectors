# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import pathlib as plib
import cv2

from tfmtcnn.prepare_data import ioutils
from tfmtcnn.models import pnet
from tfmtcnn.models import rnet
from tfmtcnn.models import onet
from tfmtcnn.mtcnn import MTCNN

imgdir = plib.Path(os.pardir, 'images').absolute()
outdir = plib.Path(os.pardir, os.pardir, os.pardir, 'results').absolute()

prefix = plib.Path(os.pardir, os.pardir, os.pardir).absolute()
prefix = [prefix.joinpath('mtcnn', 'PNet', 'pnet'),
          prefix.joinpath('mtcnn', 'RNet', 'rnet'),
          prefix.joinpath('mtcnn', 'ONet', 'onet')]

epochs = [30, 30, 30]
model_path = ['{}-{}'.format(x, y) for x, y in zip(prefix, epochs)]

mode = 'ONet'
threshold = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2


def main():
    detectors = [None, None, None]

    # load P-net model
    if mode in ('PNet', 'RNet', 'ONet'):
        detectors[0] = pnet.PNet(model_path='default')

    # load R-net model
    if mode in ('RNet', 'ONet'):
        detectors[1] = rnet.RNet(model_path='default')

    # load O-net model
    if mode in ('ONet',):
        detectors[2] = onet.ONet(model_path='default')

    detector = MTCNN(detectors=detectors,
                     min_face_size=min_face_size,
                     stride=stride,
                     threshold=threshold)

    if not outdir.exists():
        outdir.mkdir()

    loader = ioutils.ImageLoaderWithPath(os.listdir(str(imgdir)), prefix=imgdir)

    for image, path in loader:
        boxes, landmarks = detector.detect(image)

        # show rectangles
        for bbox in boxes:
            position = (int(bbox[0]), int(bbox[1]))
            cv2.putText(image, str(np.round(bbox[4], 2)), position, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            cv2.rectangle(image, position, (int(bbox[2]), int(bbox[3])), (0, 0, 255))

        # show landmarks
        for landmark in landmarks:
            for x, y in landmark:
                cv2.circle(image, (x, y), 3, (0, 0, 255))

        ioutils.write_image(image, path.name, prefix=outdir)

        cv2.imshow(str(path), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
