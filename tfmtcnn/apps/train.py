# coding:utf-8
__author__ = 'Ruslan N. Kosarev'


from pathlib import Path
import click
from object_detection.legacy import train


@click.command()
@click.option('--wider', default='~/datasets/wider', type=Path, help='Directory for Wider dataset.')
@click.option('--lfw', default='~/datasets/lfwmtcnn', type=Path, help='Directory for LFW dataset.')
@click.option('--mtcnn', default='~/models/mtcnn', type=Path, help='Directory to save trained mtcnn nets.')
def main(**args):
    pass


if __name__ == '__main__':
    main()
