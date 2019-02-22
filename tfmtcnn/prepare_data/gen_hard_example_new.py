# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
import numpy as np
from prepare_data.utils import convert_to_square
import argparse
import pickle as pickle
from train_models.mtcnn_model import PNet, R_Net, O_Net
from prepare_data.loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from prepare_data.data_utils import *
import config
from prepare_data import h5utils
from train_models import MTCNN_config
from prepare_data import ioutils


def save_hard_example(net, data, save_path, image_size):
    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    positive = []
    negative = []
    part = []

    im_idx_list = data['images']
    # print(images[0])
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    print("processing %d images in total" % num_of_images)

    # save files
    # neg_label_file = '../data/{}/neg_{}.txt'.format(net, image_size)
    # neg_file = open(neg_label_file, 'w')

    # pos_label_file = '../data/{}/pos_{}.txt'.format(net, image_size)
    # pos_file = open(pos_label_file, 'w')

    # part_label_file = '../data/{}/part_{}.txt'.format(net, image_size)
    # part_file = open(part_label_file, 'w')

    # read detect result
    det_boxes = pickle.load(open(str(save_path.joinpath('detections.pkl')), 'rb'))
    # print(len(det_boxes), num_of_images)
    print(len(det_boxes))
    print(num_of_images)
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    #im_idx_list image index(list)
    #det_boxes detect result(list)
    #gt_boxes_list gt(list)
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        #change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # Iou with all gts must below 0.3            
            if np.max(Iou) < 0.3 and neg_num < 60:
                #save the examples
                filename = neg_dir.joinpath("%s.jpg" % n_idx)
                # print(save_file)
                # neg_file.write(save_file + ' 0\n')
                ioutils.write_image(resized_im, filename)
                negative.append((os.path.join(filename.parent.name, filename.name), 0, 0, 0, 0, 0))

                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    filename = pos_dir.joinpath("%s.jpg" % p_idx)
                    ioutils.write_image(resized_im, filename)

                    positive.append((os.path.join(filename.parent.name, filename.name), 1,
                                     offset_x1, offset_y1, offset_x2, offset_y2))
                    # pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    filename = part_dir.joinpath("%s.jpg" % d_idx)
                    ioutils.write_image(resized_im, filename)

                    part.append((os.path.join(filename.parent.name, filename.name), -1,
                                 offset_x1, offset_y1, offset_x2, offset_y2))
                    # part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    d_idx += 1

    h5file = '../data/24/dbtrainrnet.h5'
    h5utils.write(h5file, 'positive', np.array(positive, dtype=MTCNN_config.wider_dtype))
    h5utils.write(h5file, 'negative', np.array(negative, dtype=MTCNN_config.wider_dtype))
    h5utils.write(h5file, 'part', np.array(part, dtype=MTCNN_config.wider_dtype))

    # neg_file.close()
    # part_file.close()
    # pos_file.close()


def t_net(config, prefix, epoch, batch_size, test_mode="PNet", thresh=(0.6, 0.6, 0.7), min_face_size=25,
          stride=2, slide_window=False, shuffle=False, vis=False):

    detectors = [None, None, None]
    print("Test model: ", test_mode)
    #PNet-echo
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

    print(model_path[0])
    print("==================================", test_mode)

    # load pnet model
    if slide_window:
        detectors[0] = Detector(PNet, 12, batch_size[0], model_path[0])
    else:
        detectors[0] = FcnDetector(PNet, model_path[0])

    # load rnet model
    if test_mode == 'RNet':
        detectors[1] = Detector(R_Net, 24, batch_size[1], model_path[1])

    # load onet model
    if test_mode == 'ONet':
        detectors[1] = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[2] = Detector(O_Net, 48, batch_size[2], model_path[2])

    basedir = '../data/'
    filename = '../data/WIDER_train/wider_face_train_bbx_gt.txt'
    data = read_annotation(basedir, filename)
    # data['images'] = data['images'][:300]
    # data['bboxes'] = data['bboxes'][:300]

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    print('load test data')
    test_data = TestLoader(data['images'])
    print('start detecting....')
    detections, landmarks = mtcnn_detector.detect_face(test_data)
    print('finish detecting ')

    save_net = 'RNet'
    if test_mode == "PNet":
        save_net = "RNet"
    elif test_mode == "RNet":
        save_net = "ONet"

    # save detect result
    save_path = data_dir.joinpath(save_net)
    print('save_path is :', save_path)
    if not save_path.exists():
        save_path.mkdir()

    save_file = save_path.joinpath(save_path, "detections.pkl")
    print(save_file)
    with open(str(save_file), 'wb') as f:
        pickle.dump(detections, f, 1)

    save_hard_example(24, data, save_path, 24)


if __name__ == '__main__':

    test_mode = 'PNet'

    if test_mode == 'PNet':
        config = config.PNetConfig()

    if test_mode == 'RNet':
        config = config.RNetConfig()

    if test_mode == 'ONet':
        config = config.ONetConfig()

    # if net == "RNet":
    #     image_size = 24
    # if net == "ONet":
    #     image_size = 48

    base_dir = plib.Path('../data/WIDER_train').absolute()
    data_dir = plib.Path('../data/').absolute().joinpath(str(24))
    
    pos_dir = data_dir.joinpath('positive')
    neg_dir = data_dir.joinpath('negative')
    part_dir = data_dir.joinpath('part')

    # create dictionary shuffle
    for path in [pos_dir, neg_dir, part_dir]:
        if not path.exists():
            path.mkdir(parents=True)

    prefix = (plib.Path(os.pardir).joinpath('mtcnn/PNet/PNet').absolute(),
              plib.Path(os.pardir).joinpath('mtcnn/RNet/RNet').absolute(),
              plib.Path(os.pardir).joinpath('mtcnn/ONet/ONet').absolute())

    epochs = (10, 14, 16)
    batch_size = (2048, 256, 16)
    thresh = (0.3, 0.1, 0.7)
    min_face = 20
    stride = 2
    slide_window = False
    shuffle = False

    print('Called with argument:')
    #print(args)
    t_net(config,
          prefix,#model param's file
          epochs, #final epoches
          batch_size, #test batch_size
          test_mode,#test which model
          thresh, #cls threshold
          min_face, #min_face
          stride,#stride
          slide_window,
          shuffle,
          vis=False)
