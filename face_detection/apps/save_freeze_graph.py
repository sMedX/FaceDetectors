"""Imports a model metagraph and checkpoint file, converts the variables to constants
and exports the model as a graphdef protobuf
"""
# MIT License
# 
# Copyright (c) 2019 sMedX

import click
import pathlib
import tensorflow as tf
import os
import re


# def get_model_filenames(model_dir):
#     model_dir = str(model_dir)
#     files = os.listdir(model_dir)
#     meta_files = [s for s in files if s.endswith('.meta')]
#     if len(meta_files)==0:
#         raise ValueError('No meta file found in the model directory (%s)' % model_dir)
#     elif len(meta_files)>1:
#         raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
#     meta_file = meta_files[0]
#     ckpt = tf.train.get_checkpoint_state(model_dir)
#     if ckpt and ckpt.model_checkpoint_path:
#         ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
#         return meta_file, ckpt_file
#
#     meta_files = [s for s in files if '.ckpt' in s]
#     max_step = -1
#     for f in files:
#         step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
#         if step_str is not None and len(step_str.groups())>=2:
#             step = int(step_str.groups()[1])
#             if step > max_step:
#                 max_step = step
#                 ckpt_file = step_str.groups()[0]
#     return meta_file, ckpt_file
#
#
# import os, argparse

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))


def save_freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
#     parser.add_argument("--output_node_names", type=str, default="",
#                         help="The name of the output nodes, comma separated.")
#     args = parser.parse_args()
#
#     freeze_graph(args.model_dir, args.output_node_names)


#@click.command()
#@click.option('--model_dir', type=pathlib.Path,
#              help='Directory with the meta graph and checkpoint files containing model parameters')
def main(**args):
    #model_dir = args['model_dir'].expanduser()

    model_dir = '/home/korus/models/detectors/ssd_inception_v2'
    output_node_names = 'image_tensor:0, detection_boxes:0, detection_scores:0'

    save_freeze_graph(model_dir, output_node_names)


if __name__ == '__main__':
    main()
