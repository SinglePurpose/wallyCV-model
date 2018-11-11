# Script for final preparation stage of the dataset.
# i.e. we pack our labels from the .csv file and our images into a single binary .tfrecord file.
# This binary file is needed for training the Object Detection Model.
# This approach is much more efficient. i.e. by storing the data in one binary file, we have it in one block
# of memory. Compared to storing each .jpg and labels separately. Which results in time-consuming read operations,
# especially on HDD drives.

import csv
import os
import tensorflow as tf
from object_detection.utils import dataset_util

image_format = b"jpg"
classes = [1]
classes_text = ["waldo".encode('utf8')]

labels_path = "PATH_TO/eval_labels.csv"
images_path = "PATH_TO/images/test"
tfrecord_output_file = "PATH_TO/eval.record"

writer = tf.python_io.TFRecordWriter(tfrecord_output_file)

with open(labels_path, newline="") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    headers = next(reader, None)
    for row in reader:
        # retrieve the info from each row in csv file
        filename = row[0]
        width = int(row[1])
        height = int(row[2])
        x_min = [float(row[4]) / width]
        y_min = [float(row[5]) / height]
        x_max = [float(row[6]) / width]
        y_max = [float(row[7]) / height]

        image_path = os.path.join(images_path, filename)
        encoded_image_data = tf.gfile.FastGFile(image_path, "rb").read()

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            "image/height": dataset_util.int64_feature(height),
            "image/width": dataset_util.int64_feature(width),
            "image/filename": dataset_util.bytes_feature(filename.encode('utf8')),
            "image/source_id": dataset_util.bytes_feature(filename.encode('utf8')),
            "image/encoded": dataset_util.bytes_feature(encoded_image_data),
            "image/format": dataset_util.bytes_feature(image_format),
            "image/object/bbox/xmin": dataset_util.float_list_feature(x_min),
            "image/object/bbox/xmax": dataset_util.float_list_feature(x_max),
            "image/object/bbox/ymin": dataset_util.float_list_feature(y_min),
            "image/object/bbox/ymax": dataset_util.float_list_feature(y_max),
            "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
            "image/object/class/label": dataset_util.int64_list_feature(classes),
        }))

        writer.write(tf_example.SerializeToString())

writer.close()
