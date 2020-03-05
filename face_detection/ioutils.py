# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import time
from functools import partial
from pathlib import Path
from PIL import Image
import numpy as np


mkdir = partial(Path.mkdir, parents=True, exist_ok=True)


def write_image(image, filename, prefix=None, mode='RGB'):
    if prefix is not None:
        filename = prefix.joinpath(filename)

    mkdir(filename.parent)

    if isinstance(image, np.ndarray):
        image = array2pil(image, mode=mode)

    if image.save(str(filename)):
        raise IOError('while writing the file {}'.format(filename))


def read_image(filename, prefix=None):
    if prefix is not None:
        image = Image.open(str(prefix.joinpath(filename)))
    else:
        image = Image.open(str(filename))

    if image is None:
        raise IOError('while reading the file {}'.format(filename))

    return image


class ImageLoader:
    def __iter__(self):
        return self

    def __init__(self, input, prefix=None, display=100, log=True):

        if not isinstance(input, (Path, list)):
            raise IOError("Input '{}' must be directory or list of files".format(input))

        if isinstance(input, list):
            self.files = input
        elif input.expanduser().is_dir():
            prefix = input.expanduser()
            self.files = [f.name for f in prefix.glob('*')]
        else:
            raise IOError("Directory '{}' does not exist".format(input))

        self.counter = 0
        self.start_time = time.time()
        self.display = display
        self.size = len(self.files)
        self.prefix = prefix
        self.log = log
        self.__filename = None

        print('Loader <{}> is initialized, number of files {}'.format(self.__class__.__name__, self.size))

    def __next__(self):
        if self.counter < self.size:
            if (self.counter + 1) % self.display == 0:
                elapsed_time = (time.time() - self.start_time) / self.display
                print('\rnumber of processed images {}/{}, {:.5f} seconds per image'.format(self.counter+1, self.size, elapsed_time), end='')
                self.start_time = time.time()

            image = read_image(self.files[self.counter], prefix=self.prefix)
            self.filename = image.filename

            if self.log:
                print('{}/{}, {}, {}'.format(self.counter, self.size, self.filename, image.size))

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
