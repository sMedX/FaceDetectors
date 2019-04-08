# coding:utf-8
__author__ = 'Ruslan N. Kosarev'
__version__ = "0.0.1"

import os
import pathlib as plib


def dirname():
    return plib.Path(os.path.dirname(__file__))


widerdir = '~/datasets/wider'
lfwdir = '~/datasets/lfwmtcnn'

threshold = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2
