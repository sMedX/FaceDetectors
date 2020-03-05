# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

from pathlib import Path
import tensorflow as tf


def load_model_from_url(model_name, base_url='http://download.tensorflow.org/models/object_detection/'):
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file, untar=True)
    return Path(model_dir)


def load_graph(file, name='frozen_inference_graph.pb'):
    if file.is_dir():
        if file.joinpath(name).is_file():
            file = file.joinpath(name)

    if not file.is_file():
        raise IOError('File {} does not exist'.format(file))

    print('Load frozen inference graph from {}'.format(file))

    with tf.gfile.GFile(str(file), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')
    return graph


