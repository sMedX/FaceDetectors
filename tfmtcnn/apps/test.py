# coding:utf-8

import os
import cv2
import numpy as np
import itertools
import pathlib as plib

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

epoch = [30, 30, 30]
model_path = ['{}-{}'.format(x, y) for x, y in zip(prefix, epoch)]

test_mode = 'ONet'
threshold = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
batch_size = [2048, 64, 16]


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

mtcnn_detector = MTCNN(detectors=detectors,
                       min_face_size=min_face_size,
                       stride=stride,
                       threshold=threshold,
                       slide_window=slide_window)


imgdir = '/home/korus/Workspace/DCNCFacialPointDetection/test/lfpw_testImage'

out_dir = imgdir + '_out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

loader = ioutils.ImageLoader(os.listdir(imgdir), prefix=imgdir)

for path, image in loader:
    print(path)
    boxes, landmarks = mtcnn_detector.detect(image)
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

    path = os.path.join(out_dir, os.path.basename(path))
    cv2.imwrite(path, image)

    cv2.imshow(path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
