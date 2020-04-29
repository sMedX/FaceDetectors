# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import cv2
import time
from functools import partial
from pathlib import Path
from PIL import Image
import numpy as np


mkdir = partial(Path.mkdir, parents=True, exist_ok=True)


def write_image(image, file, prefix=None, ext='.png'):
    file = Path(file).with_suffix(ext)
    if prefix is not None:
        file = Path(prefix).joinpath(file)

    mkdir(file.parent)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if image.save(file):
        raise IOError('while writing the file {}'.format(file))


def read_image(file, prefix=None):
    file = Path(file)
    if prefix is not None:
        file = Path(prefix).joinpath(file)

    image = Image.open(file)

    if image is None:
        raise IOError('while reading the file {}'.format(file))

    return np.asarray(image)


class VideoLoader:
    def __iter__(self):
        return self

    def __init__(self, input, prefix=None, display=100, log=True):

        if not isinstance(input, (Path, str)):
            raise IOError("Input '{}' must be file".format(input))

        self.input = Path(input)
        if prefix is not None:
            self.input = Path(prefix).joinpath(self.input)

        self._cap = cv2.VideoCapture(str(self.input))

        self.counter = 0
        self.start_time = time.time()
        self.display = display
        self.prefix = prefix
        self.log = log
        self.name = None

        print('Loader <{}> is initialized from video file {}'.format(self.__class__.__name__, self.input))

    def __next__(self):

        retval, image = self._cap.read()

        if retval:
            if self.log:
                print('{}) {}, {}'.format(self.counter, self.input, image.shape))

            self.name = '{}_{}.{}'.format(self.input.stem, self.counter, 'jpg')
            self.counter += 1

            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB )
        else:
            print('\n\rnumber of processed images {}'.format(self.counter))
            raise StopIteration

    def reset(self):
        self.counter = 0
        return self


class ImageLoader:
    def __iter__(self):
        return self

    def __init__(self, input, prefix=None, display=100, log=True):

        if not isinstance(input, (Path, list)):
            raise IOError("Input '{}' must be directory or list of files".format(input))

        if isinstance(input, list):
            self.input = input
        elif input.expanduser().is_dir():
            prefix = input.expanduser()
            self.input = [f.name for f in prefix.glob('*') if f.is_file()]
        else:
            raise IOError("Directory '{}' does not exist".format(input))

        self.counter = 0
        self.start_time = time.time()
        self.display = display
        self.size = len(self.input)
        self.prefix = prefix
        self.log = log
        self.name = None

        print('Loader <{}> is initialized, number of files {}'.format(self.__class__.__name__, self.size))

    def __next__(self):
        if self.counter < self.size:
            if (self.counter + 1) % self.display == 0:
                elapsed_time = (time.time() - self.start_time) / self.display
                print('\rnumber of processed images {}/{}, {:.5f} seconds per image'.format(self.counter+1, self.size, elapsed_time), end='')
                self.start_time = time.time()

            image = read_image(self.input[self.counter], prefix=self.prefix)
            self.name = Path(self.input[self.counter]).name

            if self.log:
                print('{}/{}, {} {}'.format(self.counter, self.size, self.name, image.shape))

            self.counter += 1
            return image
        else:
            print('\n\rnumber of processed images {}'.format(self.size))
            raise StopIteration

    def reset(self):
        self.counter = 0
        return self


def pil2array(image, mode='RGB'):
    if image.mode == mode.upper():
        return np.array(image)

    output = []

    for channel in mode.upper():
        output.append(np.array(image.getchannel(channel)))

    output = np.stack(output, axis=2)

    return output


def array2pil(image, mode='RGB'):

    default_mode = 'RGB'
    index = []

    for sym in mode.upper():
        index.append(default_mode.index(sym))

    output = Image.fromarray(image[:, :, index], mode=default_mode)

    return output


def gray_to_rgb(image: np.ndarray):
    assert len(image.shape) == 2
    return np.stack([image, image, image], axis=2)


def resize(image, size):
    size = np.asarray(size)
    if size.size == 1:
        size = np.asarray([size, size])

    image = Image.fromarray(image)
    resized = image.resize(size, Image.ANTIALIAS)

    return np.asarray(resized)

