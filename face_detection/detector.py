# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Iterable

_SRC_DIR = Path(__file__).stem


def path_to_inference(name):
    path = Path(__file__).parent.joinpath('inference', name)
    files = list(path.glob('*.pb'))

    if len(files) < 1:
        raise ValueError("Inference '{}' does not exist in the directory {}".format(name, path))

    return files[0]


def detector_names():
    names = []
    for name in Path(__file__).parent.joinpath('inference').glob('*'):
        if name.is_dir():
            names.append(str(name.name))
    names.sort()
    return names


class BoundingBox:
    def __init__(self, x, y, confidence=None):
        self.left = int(np.round(min(x)))
        self.right = int(np.round(max(x)))

        self.top = int(np.round(min(y)))
        self.bottom = int(np.round(max(y)))

        self.width = self.right - self.left
        self.height = self.bottom - self.top
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
        if self.confidence is None:
            return 'none'
        else:
            return str(np.round(self.confidence, 3))


def load_graph(filename):
    print('Load model from: {}'.format(filename))

    with tf.gfile.GFile(str(filename), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')
    return graph


class FaceDetectorBase:
    """Base FaceDetector class"""
    def __init__(self, name, margin=0.2, threshold=0.7, gpu_memory_fraction=1.0):
        self.margin = margin
        self.threshold = threshold
        self._boxes = []
        self._scores = []
        self.name = name

        self.path = path_to_inference(self.name)
        if not self.path.exists():
            raise ValueError("Inference file '{}' does not exist".format(name))

        graph = load_graph(self.path)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        self.sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')

        self.tensors = {
            'boxes': graph.get_tensor_by_name('detection_boxes:0'),
            'scores': graph.get_tensor_by_name('detection_scores:0'),
            # 'classes': graph.get_tensor_by_name('detection_classes:0'),
            # 'detections': graph.get_tensor_by_name('num_detections:0')
            }

    def get_faces(self, image: Iterable[np.ndarray]):
        """
        Performs face detection.
        Effectively on batch if GPU are used.
        Each input image should be preprocessed as as followed:
        image = FaceDetector.prepare_image(image)

        :param image:4D numpy array of float32 with (Batch, W, H, 3), RGB order
        :return: a list of size Batch, of lists, of boxes [x1, y1, x2, y2]
        """
        image = self.prepare_batch(image)

        output = self.sess.run(self.tensors, feed_dict={self.image_tensor: image})

        self.save_results(output['boxes'], output['scores'], image[0].shape)

        return self._boxes

    def get_boxes(self):
        boxes = []

        for boxes_, scores_ in zip(self._boxes, self._scores):
            boxes.append([])
            for (x1, y1, x2, y2), score in zip(boxes_, scores_):
                boxes[-1].append(BoundingBox([x1, x2], [y1, y2], confidence=score))
        return boxes

    def save_results(self, boxes_, scores_, shape):
        self._boxes = []
        self._scores = []

        for boxes, scores in zip(boxes_, scores_):
            indexes = scores > self.threshold

            boxes = boxes[indexes, :]
            scores = scores[indexes]

            self._boxes.append([])
            self._scores.append([])

            for box, score in zip(boxes, scores):
                margin = 0.5 * (box[2] - box[0]) * self.margin
                y1 = max(int(np.round((box[0] - margin) * shape[0])), 0)
                y2 = min(int(np.round((box[2] + margin) * shape[0])), shape[0])

                margin = 0.5 * (box[3] - box[1]) * self.margin
                x1 = max(int(np.round((box[1] - margin) * shape[1])), 0)
                x2 = min(int(np.round((box[3] + margin) * shape[1])), shape[1])

                # returns LEFT, TOP, RIGHT, BOTTOM
                self._boxes[-1].append([x1, y1, x2, y2])
                self._scores[-1].append(score)

    @staticmethod
    def prepare_image(image: np.ndarray) -> np.ndarray:
        """
        Make a good RGB numpy array compatible with method @prepare_batch
        if image not RGB yet.
        :param image: Image (gray or RGB)
        :return: RGB Image
        """
        # remove dim of 1, if image is greyscale
        image = np.squeeze(image)

        if image.ndim not in (2, 3):
            raise ValueError('Invalid input dimension {}'.format(image.shape))

        # transform greyscale image to rgb image
        if image.ndim == 2:
            image = ioutils.gray_to_rgb(image)

        return image

    @staticmethod
    def prepare_batch(image: Iterable[np.ndarray]):
        """
        Make a good 4D numpy array compatible with method @get_faces
        :param image: Image RGB or a batch of images of same size
        :return:
        """
        image = np.array(image)  # convert it from any iterable

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        assert len(image.shape) == 4, "Invalid input dimension"

        return image


class PYPIMTCNN(FaceDetectorBase):
    def __init__(self, gpu_memory_fraction=1.0, margin=0.2, threshold=0.7):
        super().__init__(margin=margin, threshold=threshold)
        from mtcnn.mtcnn import MTCNN
        self.__detector = MTCNN()
        self.__mode = 'RGB'

    def get_faces(self, image: Iterable[np.ndarray]):
        image = self.prepare_batch(image)

        shape = image[0].shape
        self._boxes = []
        self._scores = []

        for im in image:
            self._boxes.append([])
            self._scores.append([])

            output = self.__detector.detect_faces(im)

            for face in output:
                box = face['box']
                # The bounding box is formatted as [x, y, width, height]
                width = box[2]
                x1 = box[0]
                x2 = x1 + width

                height = box[3]
                y1 = box[1]
                y2 = y1 + height

                margin = 0.5 * width * self.margin
                x1 = max(int(np.round(x1 - margin)), 0)
                x2 = min(int(np.round(x2 + margin)), shape[1])

                margin = 0.5 * height * self.margin
                y1 = max(int(np.round(y1 - margin)), 0)
                y2 = min(int(np.round(y2 + margin)), shape[0])

                # returns LEFT, TOP, RIGHT, BOTTOM
                self._boxes[-1].append([x1, y1, x2, y2])
                self._scores[-1].append(face['confidence'])

        return self._boxes


class TFMTCNN(FaceDetectorBase):
    def __init__(self, gpu_memory_fraction=1.0, margin=0.2, threshold=0.7):
        super().__init__(margin=margin, threshold=threshold)

        from facenets.ml.detector.tfmtcnn.mtcnn import MTCNN
        self.__detector = MTCNN()
        self.__mode = 'BGR'

    def get_faces(self, image: Iterable[np.ndarray]):
        image = self.prepare_batch(image)

        shape = image[0].shape
        self._boxes = []
        self._scores = []

        for im in image:
            self._boxes.append([])
            self._scores.append([])

            im = np.stack([im[:, :, 2], im[:, :, 1], im[:, :, 0]], axis=2)
            output, _ = self.__detector.detect(im)

            if output.shape[0] < 1:
                continue

            boxes = output[:, 0:4]
            scores = output[:, 4]

            indexes = scores > self.threshold

            boxes = boxes[indexes, :]
            scores = scores[indexes]

            for box, score in zip(boxes, scores):
                margin = 0.5 * (box[2] - box[0]) * self.margin
                x1 = max(int(np.round(box[0] - margin)), 0)
                x2 = min(int(np.round(box[2] + margin)), shape[1])

                margin = 0.5 * (box[3] - box[1]) * self.margin
                y1 = max(int(np.round(box[1] - margin)), 0)
                y2 = min(int(np.round(box[3] + margin)), shape[0])

                self._boxes[-1].append([x1, y1, x2, y2])
                self._scores[-1].append(score)

        return self._boxes


class FaceDetector:
    """Abstract FaceDetector with fabric method"""

    @staticmethod
    def create(detector_name: str):
        detector = FaceDetectorBase(name=detector_name)
        return detector
