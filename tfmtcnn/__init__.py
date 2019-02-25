# coding:utf-8
__author__ = 'Ruslan N. Kosarev'
__version__ = "0.0.1"


import os
import pathlib as plib


def packagedir():
    return plib.Path(os.path.realpath(__file__)).parent
