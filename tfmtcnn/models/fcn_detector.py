
import os
import sys
import tensorflow as tf


class FcnDetector:
    def __init__(self, config, model_path):
        # create a graph
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            net = config.factory(image_reshape, training=False)
            self.cls_prob = net.cls_pro_test
            self.bbox_pred = net.bbox_pred_test
            
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()

            # check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)

            print(self.__class__.__name__)
            print(os.path.abspath(model_path))
            print(os.path.abspath(model_dict))

            readstate = ckpt and ckpt.model_checkpoint_path
            assert readstate, "the params dictionary is not valid"
            print("restore models' param")
            saver.restore(self.sess, model_path)

    def predict(self, databatch):
        height, width, _ = databatch.shape
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                            feed_dict={self.image_op: databatch,
                                                       self.width_op: width, self.height_op: height})
        return cls_prob, bbox_pred
