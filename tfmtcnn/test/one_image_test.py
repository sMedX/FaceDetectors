# coding:utf-8
import sys
import os
import cv2
import numpy as np
import itertools


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=None)


sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader

thresh = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 64, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

test_mode = 'ONet'

# load P-net model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

# load R-net model
if test_mode in ['RNet', 'ONet']:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

# load O-net model
if test_mode == 'ONet':
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

path = os.path.join(os.pardir, 'data/test/lfpw_testImage')
gt_imdb = [os.path.join(path, item) for item in os.listdir(path)]

test_data = TestLoader(gt_imdb)
# boxes, landmarks = mtcnn_detector.detect_face(test_data)

out_dir = os.path.dirname(gt_imdb[0]) + '_out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for path in gt_imdb:
    print(path)
    image = cv2.imread(path)
    boxes, landmarks = mtcnn_detector.detect(image)

    # show rectangles
    for bbox in boxes:
        position = (int(bbox[0]), int(bbox[1]))
        cv2.putText(image, str(np.round(bbox[4], 2)), position, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
        cv2.rectangle(image, position, (int(bbox[2]), int(bbox[3])), (0, 0, 255))

    # show landmarks
    for landmark in landmarks:
        for i, k in grouper(landmark, 2):
            cv2.circle(image, (i, k), 3, (0, 0, 255))

    path = os.path.join(out_dir, os.path.basename(path))
    cv2.imwrite(path, image)

    # cv2.imshow(path, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


'''
for data in test_data:
    print type(data)
    for bbox in all_boxes[0]:
        print bbox
        print (int(bbox[0]),int(bbox[1]))
        cv2.rectangle(data, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
    #print data
    cv2.imshow("lala",data)
    cv2.waitKey(0)
'''