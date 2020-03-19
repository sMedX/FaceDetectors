# coding:utf-8
# __author__ = 'Ruslan N. Kosarev

from pathlib import Path
import click
from face_detection import tfutils
import object_detection.data_decoders
from object_detection import export_inference_graph
import tensorflow as tf


# s1 = 'faster_rcnn_resnet50_coco_2018_01_28'
# name = 'ssd_inception_v2_coco_2018_01_28'
name = 'ssd_mobilenet_v2_coco_2018_03_29'
# name = 'ssd_mobilenet_v1_coco_2018_01_28'

# @click.command()
# @click.option('--wider', default='~/datasets/wider', type=Path, help='Directory for Wider dataset.')
# @click.option('--lfw', default='~/datasets/lfwmtcnn', type=Path, help='Directory for LFW dataset.')
# @click.option('--mtcnn', default='~/models/mtcnn', type=Path, help='Directory to save trained mtcnn nets.')


def main():
    model_dir = tfutils.load_model_from_url(name)
    print(model_dir)

    model = tfutils.load_frozen_graph(model_dir)
    print(model)


if __name__ == '__main__':
    main()
