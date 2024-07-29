import os
gpu_num = 3 # 制定使用的 GPU 序号，使用 "" 来启用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# 忽略来自 TensorFlow 的警告
tf.get_logger().setLevel('ERROR')

class Models(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv = Conv2D(16, (3, 3), padding='same')
        self.bn = BatchNormalization()
        self.ac = ReLU()

        self.conv2 = Conv2D(32, (3, 3), padding='same')
        self.bn2 = BatchNormalization()
        self.ac2 = ReLU()

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ac(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)

        return x


m = Models()
m.build(input_shape=(2, 8, 8, 3))
m.summary()
