# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import click
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

from face_detection import tfmtcnn
from face_detection.tfmtcnn.prepare_data import lfw, ioutils


@click.command()
@click.option('--lfw', default=tfmtcnn.lfwdir, help='path to the LFW database.')
@click.option('--show', default=False, help='show detected faces.')
def main(**args):

    # initialize loader
    dblfw = lfw.DBLFW(args['lfw'])
    print(dblfw)
    files, boxes, landmarks = dblfw.read_test_annotations()

    loader = ioutils.ImageLoaderWithPath(files, prefix=dblfw.dbasedir)

    # initialize original pypi mtcnn detector
    detector = MTCNN()

    residuals = []

    for (image, path), target_boxes, target_landmarks in zip(loader, boxes, landmarks):
        faces = detector.detect_faces(image)
        if len(faces) == 0:
            print('\nface is not detected for the image', path)
            continue

        detected_landmarks = [np.zeros([5, 2])]
        detected_landmarks[0][0] = faces[0]['keypoints']['left_eye']
        detected_landmarks[0][1] = faces[0]['keypoints']['right_eye']
        detected_landmarks[0][2] = faces[0]['keypoints']['nose']
        detected_landmarks[0][3] = faces[0]['keypoints']['mouth_left']
        detected_landmarks[0][4] = faces[0]['keypoints']['mouth_right']

        # compute inter-ocular distance
        inter_ocular = np.linalg.norm(target_landmarks[0] - target_landmarks[1])

        # compute distances between detected and target landmarks
        distances = np.linalg.norm(target_landmarks - detected_landmarks[0], axis=1)
        residuals.append(distances.mean()/inter_ocular)

        # show rectangles
        if args['show']:
            for face in faces:
                # the bounding box is formatted as [x, y, width, height]
                bbox = face['box']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]

                position = (int(bbox[0]), int(bbox[1]))
                cv2.rectangle(image, position, (int(bbox[2]), int(bbox[3])), color=(0, 0, 255))

                text = str(np.round(face['confidence'], 2))
                cv2.putText(image, text, position, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))

            # show detected landmarks
            for landmark in detected_landmarks:
                for x, y in landmark:
                    cv2.circle(image, (np.float32(x), np.float32(y)), 3, (0, 0, 255))

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
