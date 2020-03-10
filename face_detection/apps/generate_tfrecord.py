"""
Usage:

# Create train data:
python generate_tfrecord.py --label=<LABEL> --txt_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python generate_tfrecord.py --label=<LABEL> --txt_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record
"""

import os
import io
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

flags = tf.app.flags
flags.DEFINE_string('txt_input', '', 'Path to the txt input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# if your image has more labels input them as
# flags.DEFINE_string('label0', '', 'Name of class[0] label')
# flags.DEFINE_string('label1', '', 'Name of class[1] label')
# and so on.
flags.DEFINE_string('img_path', '', 'Path to images')
flags.DEFINE_boolean('remove_invalid', False, 'Remove or not faces with attribute `invalid`')
flags.DEFINE_integer('minsize', 0, 'Skip faces with less size')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
# for multiple labels add more else if statements
def class_text_to_int(row_label):
    return 1
    # comment upper if statement and uncomment these statements for multiple labelling
    # if row_label == FLAGS.label0:
    #   return 1
    # elif row_label == FLAGS.label1:
    #   return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(input_stream):
    line = input_stream.readline()
    if line is '':
        return None

    path = line[:-1]
    path = os.path.join(FLAGS.img_path, path)
    with tf.gfile.GFile(path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = path.encode('utf8')
    image_format = b'jpg'

    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    class_txt = 'face'

    nobjects = int(input_stream.readline()[:-1])

    for i in range(nobjects):
        row_split = input_stream.readline()[:-1].split(' ')[:-1]
        row_split = list(map(lambda s: int(s), row_split))
        invalid = row_split[-3]

        if invalid and FLAGS.remove_invalid:
            continue

        xmin, ymin, xsize, ysize = tuple(row_split[:4])

        if xsize < FLAGS.minsize:
            continue

        xmax = xmin + xsize
        ymax = ymin + ysize

        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)

        classes_text.append(class_txt.encode('utf8'))
        classes.append(class_text_to_int(class_txt))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    count = 0
    with open(FLAGS.txt_input) as f:
        while 1:
            tf_example = create_tf_example(f)
            if tf_example is None:
                break
            writer.write(tf_example.SerializeToString())
            count += 1

    writer.close()
    print('\n')
    print('Successfully created the TFRecords: {}'.format(FLAGS.output_path))
    print('Number of tf examples: {}'.format(count))


if __name__ == '__main__':
    tf.app.run()
