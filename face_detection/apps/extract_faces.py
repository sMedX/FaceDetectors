"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 Ruslan N. Kosarev

import os
import numpy as np
from time import sleep
import random
import click

from face_detection import ioutils
from face_detection.tfmtcnn.face_detector import FaceDetector


@click.command()
@click.option('--input', help='Directory with unaligned images.')
@click.option('--output', help='Directory to save extracted faces.')
@click.option('--image_size', default=182, help='Image size (height, width) in pixels.')
@click.option('--margin', default=44, help='Margin for the crop around the bounding box (height, width) in pixels.')
@click.option('--detector', default='frcnnv3', help='type of face detector, pypimtcnn, tfmtcnn, or frcnnv3')
@click.option('--detect_multiple_faces', default=False, help='Detect and align multiple faces per image.')
@click.option('--gpu_memory_fraction', default=1.0,
              help='Upper bound on the amount of GPU memory that will be used by the process.')
def main(**args):
    sleep(random.random())

    input_dir = os.path.expanduser(args['input'])
    output_dir = os.path.expanduser(args['output'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loader = ioutils.ImageLoader(input_dir)

    # Store some git revision info in a text file in the log directory
    #src_path, _ = os.path.split(os.path.realpath(__file__))
    #facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    #dataset = facenet.get_dataset(args.input_dir)

    print('Creating detector')
    detector = FaceDetector(detector=args['detector'])

    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_{}.txt'.format(random_key))

    with open(bounding_boxes_filename, 'w') as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0

        for img in loader:
            pass

        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)


            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:

                        bounding_boxes, _ = detector.detect(img)
                        nrof_faces = bounding_boxes.shape[0]

                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]

                            if nrof_faces > 1:
                                if args['detect_multiple_faces']:
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                    index = np.argmax(
                                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                    det_arr.append(det[index, :])
                            else:
                                det_arr.append(np.squeeze(det))

                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                                bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                                bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                                bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
                                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                                scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)

                                if args['detect_multiple_faces']:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)

                                ioutils.write_image(scaled, output_filename_n)
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('{}\n'.format(output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


if __name__ == '__main__':
    main()
