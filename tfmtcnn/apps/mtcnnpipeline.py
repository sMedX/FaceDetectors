# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from prepare_data import wider, lfw, tfrecords, examples
from train_models import PNet, RNet, ONet
from train_models.train import train


# default directory to save train data
basedir = plib.Path(os.pardir).joinpath('dbase').absolute()

# default directory to save trained nets
netdir = plib.Path(os.pardir).joinpath('mtcnn').absolute()


class DBNet:
    def __init__(self, basedir, dirname='PNet', label='pnet'):
        self.output = basedir.joinpath(dirname).absolute()
        self.h5file = self.output.joinpath(label + '.h5')
        self.tfprefix = self.output.joinpath(label)


def main():
    # seed to initialize random generator.
    seed = None

    # config for input wider and lfw data
    dbwider = wider.DBWider(basedir.joinpath('wider'))
    dblfw = lfw.DBLFW(basedir.joinpath('lfw'))

    # initialize config for datasets and nets
    pnet = PNet.Config()
    pnet.dbase = DBNet(basedir, dirname='PNet', label='pnet')
    pnet.prefix = netdir.joinpath('PNet', 'pnet')

    rnet = RNet.Config()
    rnet.dbase = DBNet(basedir, dirname='RNet', label='rnet')
    rnet.prefix = netdir.joinpath('RNet', 'rnet')

    onet = ONet.Config()
    onet.dbase = DBNet(basedir, dirname='ONet', label='onet')
    onet.prefix = netdir.joinpath('ONet', 'onet')

    # ------------------------------------------------------------------------------------------------------------------
    # train P-Net (prediction net)

    # # prepare train data
    # wider.prepare(dbwider, pnet.dbase, image_size=pnet.image_size, seed=seed)
    # lfw.prepare(dblfw, pnet.dbase, image_size=pnet.image_size, seed=seed)
    #
    # # save tf record files
    # tfrecords.write_multi_tfrecords(pnet.dbase.h5file, prefix=pnet.dbase.tfprefix, seed=seed)
    #
    # # train
    # train(pnet, tfprefix=pnet.dbase.tfprefix, prefix=pnet.prefix, seed=seed)
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # train R-Net (refinement net)
    #
    # # prepare train data
    # examples.generate(dbwider, models=(pnet, rnet), threshold=(0.3, 0.1, 0.7), min_face_size=20, stride=2)
    # lfw.prepare(dblfw, rnet.dbase, image_size=rnet.image_size, seed=seed)
    #
    # # save tf record files
    # tfrecords.write_multi_tfrecords(rnet.dbase.h5file, prefix=rnet.dbase.tfprefix, seed=seed)
    #
    # # train
    # train(rnet, tfprefix=rnet.dbase.tfprefix, prefix=rnet.prefix, seed=seed)
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # train O-Net (refinement net)
    #
    # # prepare train data
    # examples.generate(dbwider, models=(pnet, rnet, onet), threshold=(0.3, 0.1, 0.7), min_face_size=20, stride=2)
    # lfw.prepare(dblfw, onet.dbase, image_size=onet.image_size, seed=seed)
    #
    # # save tf record files
    # tfrecords.write_multi_tfrecords(onet.dbase.h5file, prefix=onet.dbase.tfprefix, seed=seed)

    # train
    train(onet, tfprefix=onet.dbase.tfprefix, prefix=onet.prefix, seed=seed)


if __name__ == '__main__':
    main()