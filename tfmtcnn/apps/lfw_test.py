# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import click
import cv2
import numpy as np

import tfmtcnn
from tfmtcnn.prepare_data import ioutils, lfw
from tfmtcnn.mtcnn import MTCNN


@click.command()
@click.option('--lfw', default=tfmtcnn.lfwdir, help='path to the LFW database.')
@click.option('--show', default=False, help='show detected faces.')
def main(**args):

    # initialize loader
    dblfw = lfw.DBLFW(args['lfw'])
    print(dblfw)
    files, boxes, landmarks = dblfw.read_test_annotations()

    loader = ioutils.ImageLoaderWithPath(files, prefix=dblfw.dbasedir)

    # initialize detector
    detector = MTCNN(min_face_size=tfmtcnn.min_face_size,
                     stride=tfmtcnn.stride,
                     threshold=tfmtcnn.threshold)

    residuals = []

    for (image, path), target_boxes, target_landmarks in zip(loader, boxes, landmarks):
        detected_boxes, detected_landmarks = detector.detect(image)

        if len(detected_landmarks) == 0:
            print('\nface is not detected for the image', path)
            continue

        # compute inter-ocular distance
        inter_ocular = np.linalg.norm(target_landmarks[0] - target_landmarks[1])

        # compute distances between detected and target landmarks
        distances = np.linalg.norm(target_landmarks - detected_landmarks[0], axis=1)

        residuals.append(distances.mean()/inter_ocular)

        # show rectangles
        if args['show']:
            for bbox in detected_boxes:
                position = (int(bbox[0]), int(bbox[1]))
                cv2.putText(image, str(np.round(bbox[4], 2)), position, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
                cv2.rectangle(image, position, (int(bbox[2]), int(bbox[3])), color=(0, 0, 255))

            # show detected landmarks
            for landmark in detected_landmarks:
                for x, y in landmark:
                    cv2.circle(image, (x, y), 3, (0, 0, 255))

            # show target landmarks
            for x, y in target_landmarks:
                cv2.circle(image, (np.float32(x), np.float32(y)), 3, (255, 0, 0))

            cv2.imshow(str(path), image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print('errors landmark detection for LFW database')
    print('    accuracy', len(residuals)/loader.size)
    print('median error', np.median(residuals))
    print('  mean error', np.mean(residuals))


if __name__ == '__main__':
    main()
