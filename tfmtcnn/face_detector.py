# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import numpy as np
from evaluface.ioutils import pil2array


class BoundingBox:
    def __init__(self, left, top, width, height, confidence=None):
        self.left = int(np.round(left))
        self.top = int(np.round(top))

        self.right = int(np.round(left + width)) + 1
        self.bottom = int(np.round(top + height)) + 1

        self.width = self.right - self.left - 1
        self.height = self.bottom - self.top - 1
        self.confidence = confidence

    def __repr__(self):
        return 'x = {}, y = {}, width = {}, height = {}, confidence = {}'.format(self.left, self.top, self.width, self.height, self.confidence)

    @property
    def left_upper(self):
        return self.left, self.top

    @property
    def right_lower(self):
        return self.right, self.bottom

    @property
    def confidence_as_string(self):
        return str(np.round(self.confidence, 3))


class MTCNN:
    def __init__(self):
        from mtcnn.mtcnn import MTCNN
        self.__detector = MTCNN().detect_faces
        self.__mode = 'BGR'

    def detector(self, image):
        faces = self.__detector(pil2array(image, self.__mode))
        bboxes = []

        for face in faces:
            box = face['box']
            bbox = BoundingBox(left=box[0], top=box[1], width=box[2], height=box[3], confidence=face['confidence'])
            bboxes.append(bbox)

        return bboxes


class TFMTCNN:
    def __init__(self):
        from evaluface.detectors.tfmtcnn.mtcnn import MTCNN
        self.__detector = MTCNN().detect
        self.__mode = 'BGR'

    def detector(self, image):
        boxes, _ = self.__detector(pil2array(image, self.__mode))

        bboxes = []

        for box in boxes:
            bbox = BoundingBox(left=box[0], top=box[1], width=box[2] - box[0], height=box[3] - box[1], confidence=box[4])
            bboxes.append(bbox)

        return bboxes


class FasterRCNNv3:
    def __init__(self):
        from evaluface.detectors.frcnnv3 import detector
        self.__detector = detector.FaceDetector().get_faces
        self.__mode = 'RGB'

    def detector(self, image):

        boxes, scores = self.__detector(pil2array(image, self.__mode))
        bboxes = []

        for (y1, x1, y2, x2), score in zip(boxes, scores):
            bbox = BoundingBox(left=x1, top=y1, width=x2-x1, height=y2-y1, confidence=score)
            bboxes.append(bbox)

        return bboxes


class FaceDetector:
    def __init__(self, detector='tfmtcnn'):

        if detector is 'pypimtcnn':
            self.__detector = MTCNN().detector

        elif detector is 'tfmtcnn':
            self.__detector = TFMTCNN().detector

        elif detector is 'frcnnv3':
            self.__detector = FasterRCNNv3().detector

        else:
            raise 'Undefined face detector type {}'.format(detector)

    def detect(self, image):
        return self.__detector(image)
