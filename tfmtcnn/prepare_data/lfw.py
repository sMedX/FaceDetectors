# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import cv2
import numpy as np
import h5py
from prepare_data.utils import IoU, readlines
from prepare_data.BBox_utils import read_bbox_data, BBox
from prepare_data.Landmark_utils import rotate, flip
from prepare_data import h5utils
from prepare_data import ioutils

dtype = np.dtype([('path', h5py.special_dtype(vlen=str)), ('label', np.int8),
                  ('1', np.float), ('2', np.float), ('3', np.float), ('4', np.float), ('5', np.float),
                  ('6', np.float), ('7', np.float), ('8', np.float), ('9', np.float), ('10', np.float)])


class DBLFW:
    def __init__(self, path):
        self.path = path
        self.images = path.joinpath('')
        self.annotations = path.joinpath('trainImageList.txt')
        self.label = 'landmark'


def prepare(dbase, outdbase, image_size=12, augment=True, seed=None):
    np.random.seed(seed=seed)

    outdir = outdbase.output.joinpath(dbase.label)
    if not outdir.exists():
        outdir.mkdir()

    data = read_bbox_data(dbase.annotations)

    files = []
    list_of_boxes = []
    list_of_landmarks = []

    for sample in data:
        files.append(sample[0])
        list_of_boxes.append(sample[1])
        list_of_landmarks.append(sample[2])

    loader = ioutils.ImageLoader(files, prefix=dbase.images)

    idx = 0
    image_id = 0
    output = []

    # image_path bbox landmark (5*2)
    for img, bbox, landmarkGt in zip(loader, list_of_boxes, list_of_landmarks):

        # for (inpfile, bbox, landmarkGt) in data:
        # img = ioutils.read_image(inpfile, prefix=dbase.images)
        height, width, channel = img.shape

        f_imgs = []
        f_landmarks = []

        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        # get sub-image from bbox
        f_face = img[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]

        # resize the gt image to specified size
        f_face = cv2.resize(f_face, (image_size, image_size))
        # initialize the landmark
        landmark = np.zeros((5, 2))

        # normalize land mark by dividing the width and height of the ground truth bounding box
        # landmakrGt is a list of tuples
        for index, one in enumerate(landmarkGt):
            # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
            landmark[index] = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]),
                               (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))

        f_imgs.append(f_face)
        f_landmarks.append(landmark.reshape(10))
        landmark = np.zeros((5, 2))

        if augment:
            idx = idx + 1
            # if idx % 100 == 0:
            #     print('\r{} images have been processed'.format(idx), end='')

            x1, y1, x2, y2 = gt_box
            # gt's width
            gt_w = x2 - x1 + 1
            # gt's height
            gt_h = y2 - y1 + 1
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            # random shift
            for i in range(10):
                bbox_size = np.random.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = np.random.randint(-0.2*gt_w, 0.2*gt_w)
                delta_y = np.random.randint(-0.2*gt_h, 0.2*gt_h)
                nx1 = int(max(x1 + gt_w / 2 - bbox_size / 2 + delta_x, 0))
                ny1 = int(max(y1 + gt_h / 2 - bbox_size / 2 + delta_y, 0))

                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
                resized_im = cv2.resize(cropped_im, (image_size, image_size))
                # cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box, 0))

                if iou > 0.65:
                    f_imgs.append(resized_im)
                    # normalize
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0] - nx1) / bbox_size, (one[1] - ny1) / bbox_size)
                        landmark[index] = rv
                    f_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = f_landmarks[-1].reshape(-1, 2)
                    bbox = BBox([nx1, ny1, nx2, ny2])

                    # mirror
                    if np.random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (image_size, image_size))
                        # c*h*w
                        f_imgs.append(face_flipped)
                        f_landmarks.append(landmark_flipped.reshape(10))
                    # rotate
                    if np.random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, bbox.reprojectLandmark(landmark_), 5)
                        # landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (image_size, image_size))
                        f_imgs.append(face_rotated_by_alpha)
                        f_landmarks.append(landmark_rotated.reshape(10))

                        # flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (image_size, image_size))
                        f_imgs.append(face_flipped)
                        f_landmarks.append(landmark_flipped.reshape(10))

                        # anti-clockwise rotation
                    if np.random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, bbox.reprojectLandmark(landmark_), -5)
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (image_size, image_size))
                        f_imgs.append(face_rotated_by_alpha)
                        f_landmarks.append(landmark_rotated.reshape(10))

                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (image_size, image_size))
                        f_imgs.append(face_flipped)
                        f_landmarks.append(landmark_flipped.reshape(10))

            f_imgs, f_landmarks = np.asarray(f_imgs), np.asarray(f_landmarks)

            for i in range(len(f_imgs)):
                if np.sum(np.where(f_landmarks[i] <= 0, 1, 0)) > 0:
                    continue

                if np.sum(np.where(f_landmarks[i] >= 1, 1, 0)) > 0:
                    continue

                key = os.path.join(dbase.label, '{}.jpg'.format(image_id))
                ioutils.write_image(f_imgs[i], key, prefix=outdbase.output)
                output.append(tuple([key, -2] + f_landmarks[i].tolist()))

                image_id += 1

    print('\r{} images have been processed'.format(idx))
    h5utils.write(outdbase.h5file, dbase.label, np.array(output, dtype=dtype))
