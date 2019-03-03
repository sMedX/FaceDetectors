# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import cv2
import time
import pathlib as plib


def write_image(image, filename, prefix=None):
    if prefix is not None:
        filename = os.path.join(str(prefix), str(filename))

    if not cv2.imwrite(str(filename), image):
        raise IOError('while writing the file {}'.format(filename))


def read_image(filename, prefix=None):
    if prefix is not None:
        image = cv2.imread(os.path.join(str(prefix), str(filename)))
    else:
        image = cv2.imread(str(filename))

    if image is None:
        raise IOError('while reading the file {}'.format(filename))
    return image


class ImageLoader:
    def __iter__(self):
        return self

    def __init__(self, data, prefix=None, display=100):
        self.counter = 0
        self.start_time = time.time()
        self.data = data
        self.display = display
        self.size = len(data)
        self.prefix = str(prefix)

        print('Loader <{}> is initialized, number of images {}'.format(self.__class__.__name__, self.size))

    def __next__(self):
        if self.counter < self.size:
            if (self.counter + 1) % self.display == 0:
                elapsed_time = (time.time() - self.start_time) / self.display
                print('\rnumber of processed images {}/{}, {:.5f} seconds per image'.
                      format(self.counter+1, self.size, elapsed_time), end='')
                self.start_time = time.time()

            image = read_image(self.data[self.counter], prefix=self.prefix)

            self.counter += 1
            return image
        else:
            print('\rnumber of processed images {}'.format(self.size))
            raise StopIteration

    def reset(self):
        self.counter = 0
        return self


class ImageLoaderWithPath:
    def __iter__(self):
        return self

    def __init__(self, data, prefix=None, display=100):
        self.counter = 0
        self.start_time = time.time()
        self.data = data
        self.display = display
        self.size = len(data)
        self.prefix = str(prefix)

        print('Loader <{}> is initialized, number of images {}'.format(self.__class__.__name__, self.size))

    def __next__(self):
        if self.counter < self.size:
            if (self.counter + 1) % self.display == 0:
                elapsed_time = (time.time() - self.start_time) / self.display
                print('\rnumber of processed images {}/{}, {:.5f} seconds per image'.
                      format(self.counter+1, self.size, elapsed_time), end='')
                self.start_time = time.time()

            path = plib.Path(os.path.join(str(self.prefix), self.data[self.counter]))
            image = read_image(path)

            self.counter += 1

            return image, path
        else:
            print('\rnumber of processed images {}'.format(self.size))
            raise StopIteration

    def reset(self):
        self.counter = 0
        return self
