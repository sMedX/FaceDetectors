# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from tfmtcnn.prepare_data import wider, lfw, tfrecords, examples
from tfmtcnn.models import pnet, rnet, onet
from tfmtcnn.models.train import train


# default directory to save train data
dbasedir = plib.Path(os.pardir, 'dbase').absolute()

# default directory to save trained nets
mtcnndir = plib.Path(os.pardir, 'mtcnn').absolute()

threshold = (0.6, 0.7, 0.7)
min_face_size = 20
stride = 2


class DBNet:
    def __init__(self, basedir, dirname='PNet', label='pnet'):
        self.output = basedir.joinpath(dirname).absolute()
        self.h5file = self.output.joinpath(label + '.h5')
        self.tfprefix = self.output.joinpath(label)


def main():
    # seed to initialize random generator.
    seed = None

    # config for input wider and lfw data
    dbwider = wider.DBWider(dbasedir.joinpath('wider'))
    dblfw = lfw.DBLFW(dbasedir.joinpath('lfw'))

    nets = [None, None, None]

    # initialize config for data sets and nets
    config = pnet.Config()
    config.dbase = DBNet(dbasedir, dirname='PNet', label='pnet')
    config.prefix = mtcnndir.joinpath('PNet', 'pnet')
    nets[0] = pnet.PNet(config=config)

    config = rnet.Config()
    config.dbase = DBNet(dbasedir, dirname='RNet', label='rnet')
    config.prefix = mtcnndir.joinpath('RNet', 'rnet')
    nets[1] = rnet.RNet(config=config)

    config = onet.Config()
    config.dbase = DBNet(dbasedir, dirname='ONet', label='onet')
    config.prefix = mtcnndir.joinpath('ONet', 'onet')
    nets[2] = onet.ONet(config=config)

    # ------------------------------------------------------------------------------------------------------------------
    # train P-Net (prediction net)

    # prepare train data
    net = nets[0]
    config = net.config

    wider.prepare(dbwider, config.dbase, image_size=config.image_size, seed=seed)
    lfw.prepare(dblfw, config.dbase, image_size=config.image_size, seed=seed)

    # save tf record files
    tfrecords.write_multi_tfrecords(config.dbase.h5file, prefix=config.dbase.tfprefix, seed=seed)

    # train
    train(net, tfprefix=config.dbase.tfprefix, prefix=config.prefix, seed=seed)

    # ------------------------------------------------------------------------------------------------------------------
    # train R-Net (refinement net)
    net = nets[1]
    config = net.config

    # prepare train data
    examples.generate(dbwider, models=(nets[0].config, nets[1].config), threshold=threshold, min_face_size=min_face_size, stride=stride)
    lfw.prepare(dblfw, config.dbase, image_size=config.image_size, seed=seed)

    # save tf record files
    tfrecords.write_multi_tfrecords(config.dbase.h5file, prefix=config.dbase.tfprefix, seed=seed)

    # train
    train(net, tfprefix=config.dbase.tfprefix, prefix=config.prefix, seed=seed)

    # ------------------------------------------------------------------------------------------------------------------
    # train O-Net (refinement net)
    net = nets[2]
    config = net.config

    # prepare train data
    examples.generate(dbwider, models=(nets[0].config, nets[1].config, nets[2].config), threshold=threshold, min_face_size=20, stride=2)
    lfw.prepare(dblfw, config.dbase, image_size=config.image_size, seed=seed)

    # save tf record files
    tfrecords.write_multi_tfrecords(config.dbase.h5file, prefix=config.dbase.tfprefix, seed=seed)

    # train
    train(net, tfprefix=config.dbase.tfprefix, prefix=config.prefix, seed=seed)


if __name__ == '__main__':
    main()
