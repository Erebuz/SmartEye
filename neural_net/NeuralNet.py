import os
from neural_net.yolov3 import YoloV3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


class NeuralNetwork:
    def __init__(self):
        self.size = 416
        neural_net = YoloV3()

        self.model = neural_net.get_model()
        self.class_names = neural_net.get_class_names()

    @staticmethod
    def _preprocess_image(x_train, size):
        return (tf.image.resize(x_train, (size, size))) / 255

    def detect_objects(self, image, white_list=None):
        img_dim4 = tf.expand_dims(image, 0)
        img_dim4 = self._preprocess_image(img_dim4, self.size)

        return self.model(img_dim4)
