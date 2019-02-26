# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import pathlib as plib
import cv2

from tfmtcnn.prepare_data import ioutils
from tfmtcnn.models.detector import Detector
from tfmtcnn.models.fcn_detector import FcnDetector
from tfmtcnn.models import PNet as pnet
from tfmtcnn.models import RNet as rnet
from tfmtcnn.models import ONet as onet
from tfmtcnn.mtcnn import MTCNN

imgdir = plib.Path(os.pardir, 'images').absolute()
outdir = plib.Path(os.pardir, os.pardir, os.pardir, 'results').absolute()

prefix = plib.Path(os.pardir, os.pardir, os.pardir).absolute()
prefix = [prefix.joinpath('mtcnn', 'PNet', 'pnet'),
          prefix.joinpath('mtcnn', 'RNet', 'rnet'),
          prefix.joinpath('mtcnn', 'ONet', 'onet')]

epochs = [30, 30, 30]
model_path = ['{}-{}'.format(x, y) for x, y in zip(prefix, epochs)]

test_mode = 'ONet'
threshold = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2
slide_window = False
batch_size = [2048, 64, 16]


def main():

    detectors = [None, None, None]

    # load P-net model
    if slide_window:
        detectors[0] = Detector(pnet.Config(), batch_size[0], model_path[0])
    else:
        detectors[0] = FcnDetector(pnet.Config(), model_path[0])

    # load R-net model
    if test_mode in ('RNet', 'ONet'):
        detectors[1] = Detector(rnet.Config(), batch_size[1], model_path[1])

    # load O-net model
    if test_mode is 'ONet':
        detectors[2] = Detector(onet.Config(), batch_size[2], model_path[2])

    detector = MTCNN(detectors=detectors,
                     min_face_size=min_face_size,
                     stride=stride,
                     threshold=threshold,
                     slide_window=slide_window)

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
