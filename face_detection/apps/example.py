# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import click
import cv2
import numpy as np
from pathlib import Path

from face_detection import ioutils, config
from face_detection.detector import FaceDetector


@click.command()
@click.option('--detector', default=config.ssd_inception_v2_coco,
              help='type of face detector, pypimtcnn, tfmtcnn, or others')
@click.option('--input', default=config.dir_images, type=Path,
              help='video file or directory to read images')
@click.option('--output', default=config.dir_output, type=Path,
              help='directory to save processed images with frames')
@click.option('--show', default=1, help='show face detections')
def main(**args):
    """Simple program to detect faces with tfmtcnn and pypi mtcnn detectors."""

    detector = FaceDetector.create(args['detector'])

    if args['input'].is_file():
        loader = ioutils.VideoLoader(args['input'])
    else:
        loader = ioutils.ImageLoader(args['input'])

    for image in loader:
        boxes = detector.get_faces(np.expand_dims(image, axis=0))
        print('number of detected faces', len(boxes[0]))
        print(boxes)

        # show rectangles
        image_ = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes = detector.get_boxes()

        for bbox in boxes[0]:
            print(bbox)

            # draw text and frames
            cv2.putText(image_, bbox.confidence_as_string, bbox.left_upper, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            cv2.rectangle(image_, bbox.left_upper, bbox.right_lower, (0, 0, 255))

        ioutils.write_image(ioutils.array2pil(image_, mode='BGR'),
                            loader.name, prefix=args['output'], )

        if args['show']:
            cv2.imshow(loader.name, image_)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
