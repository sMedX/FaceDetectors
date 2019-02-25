# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import pathlib as plib
import cv2

from prepare_data import ioutils
from models.detector import Detector
from models.fcn_detector import FcnDetector
from models import PNet as pnet
from models import RNet as rnet
from models import ONet as onet
from mtcnn import MTCNN

basedir = plib.Path().joinpath(os.pardir, os.pardir).absolute()
prefix = [basedir.joinpath('mtcnn', 'PNet', 'pnet'),
          basedir.joinpath('mtcnn', 'RNet', 'rnet'),
          basedir.joinpath('mtcnn', 'ONet', 'onet')]

imgdir = plib.Path('images').absolute()
outdir = plib.Path(os.pardir).joinpath(os.pardir, 'results').absolute()

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
    if test_mode in ['RNet', 'ONet']:
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

    loader = ioutils.ImageLoader(os.listdir(str(imgdir)), prefix=imgdir)

    for path, image in loader:
        print(path)
        boxes, landmarks = detector.detect(image)
        print(landmarks.shape)

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
