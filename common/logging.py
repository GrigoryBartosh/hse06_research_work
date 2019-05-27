import os
import datetime

import tensorflow as tf

from common.config import PATH

__all__ = ["TensorBoard"]


class TensorBoard():
    def __init__(self):
        name = str(datetime.datetime.now())[:19]
        logs_path = os.path.join(PATH["TF_LOGS"], name)
        self.tf_writer = tf.summary.FileWriter(logs_path)

        self.tag_count = {}

    def add_to_graph(self, tag, val):
        summary = tf.Summary(
            value=[
                tf.Summary.Value(
                    tag=tag, 
                    simple_value=val
                )
            ])

        step = self.tag_count.get(tag, 0)
        self.tag_count[tag] = step + 1

        self.tf_writer.add_summary(summary, step)
