r'''
测试 Sionna 感知场景目标移动
'''

import os
gpu_num = 3 # 指定使用的 GPU 序号，使用 "" 来启用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from sionna_sensing.single_target_sensing.sensing_env import Environment
from sionna_sensing.configs.config_load import config_load

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# 忽略来自 TensorFlow 的警告
tf.get_logger().setLevel('ERROR')

tf.random.set_seed(1) # 设置全局随机种子以保证可复现性

def motion_test():
    for step in range(3000):
        print(f"====={step}th episode start=====")
        env.reset()                         # 重置环境
        inner_step = 0
        while True:
            print(f"\r【{step}-{inner_step}th step】")
            for idx,target in enumerate(env.targets):
                _,_,env.done[idx] = target.move(env.DASX,env.DASY,env.TIME_SLOT)
                print(f"\ttarget_{idx} pos:[{float(target.pos_now[0]):-6.1f},{float(target.pos_now[1]):-6.1f},{float(target.pos_now[2]):-6.2f}]\tv:{np.linalg.norm(target.velocity_now):.2f}")
            inner_step += 1
            if all(env.done): # 所有目标到达终点
                break

if __name__ == "__main__":
    np.set_printoptions(precision=1)                           # 设置打印精度为一位小数
    config_name = 'config'                                     # 配置文件名
    
    config_path = f'./sionna_sensing/configs/{config_name}.json'    # 配置文件路径
    env_configs, target_configs = config_load(config_path)          # 读取配置文件
    scene_name = env_configs['scene_name']                          # 场景名称
    scene_training_path = f'./DL_workspace/{scene_name}'            # 场景模型训练工作区路径
    image_save_path = f'{scene_training_path}/images/'              # 场景图像保存路径
    env = Environment(image_save_path,target_configs,**env_configs) # 初始化环境
    
    motion_test()