r'''
测试 GPU 是否能完成 Sionna 加载场景
'''

import os
gpu_num = 3 # 指定使用的 GPU 序号，使用 "" 来启用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import sionna
from sionna.rt import load_scene
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

if tf.config.list_physical_devices('GPU'):
    print("GPU可用")
else:
    print("GPU不可用")

if __name__ =="__main__":
    
    scene = load_scene(sionna.rt.scene.munich)
