import os
gpu_num = 0 # 指定使用的 GPU 序号，使用 "" 来启用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
from shutil import copyfile
from sionna_sensing.multi_targets_sensing.sensing_env_multi_agent import Environment
from sionna_sensing.config.config_load import config_load

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

# 需要修改
ARG_LIST = ['configuration', 'learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency',
            'maximum_exploration', 'first_step_memory', 'replay_steps', 'number_nodes']

def set_args():
    r'''
    设置参数，可通过命令行更改
    
    Output
    -----
    vars(parser.parse_args()): dict[str, Any]
        参数字典
    '''
    
    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configuration', default='config test', type=str, help='配置文件名称, Name of configuration file')
    parser.add_argument('-e', '--episode-number', default=3000, type=int, help='训练代数，Number of episodes')
    parser.add_argument('-l', '--learning-rate', default=0.9, type=float, help='学习率，Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='Adam', help='优化方法，Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=1000, type=int, help='内存容量，Memory capacity')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='批次大小，Batch size')
    parser.add_argument('-t', '--target-frequency', default=100, type=int,
                        help='目标网络更新步数间隔，Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=3000, type=int, help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory', default=0, type=float,
                        help='填满内存的初始步骤数，Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=4, type=float, help='更新评估网络的步数间隔，steps between updating the network')
    parser.add_argument('-nn', '--number-nodes', default=512, type=int, help='神经网络每层的节点数，Number of nodes in each layer of NN')

    parser.add_argument('-gn', '--gpu-num', default='2', type=str, help='训练使用的 GPU 编号, GPU number used for training')
    parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase if "store_false"')
    
    return vars(parser.parse_args())
  
def test():
    total_step = 0
    reward_sum_his = [] # 历史累积奖励
    max_score = -10000
    
    reward_sum = 0          # 记录每个episode的累计奖励
    time_step = 0
    state = env.reset()
    # state: [target_num, 2]
    print(f'state:{state}(shape:{np.shape(state)})')
    targets_info = "\t"
    for idx, target in enumerate(env.targets):
        targets_info += f"target_{idx}[({float(target.pos_now[0]):.1f},{float(target.pos_now[1]):.1f},{float(target.pos_now[2]):.2f}),{np.linalg.norm(target.velocity_now):.2f}] "
    print(targets_info)

    actions = []
    action_types = []
    actions_info = f"\tactions:"
    # 执行动作
    for idx, agent in enumerate(agents):
        action, action_type = agent.choose_action(state)
        actions.append(action)
        action_types.append(action_type)
        if idx != len(agents)-1:
            actions_info += f'{action}({action_type}),'
        else:
            actions_info += f'{action}({action_type})'
    state_, reward, done = env.step(actions)   # 更新环境，获取环境信息
    print(actions_info+f'\treward:{reward:.4f}')
    for agent in agents:
        # 获取全局观测
        agent.observe((state, actions, reward, state_, done))
        agent.decay_epsilon()
        # print(agent.memory.memory)
        agent.replay()
        agent.update_target_model()

    total_step += 1
    time_step += 1
    state = state_
    reward_sum += reward # 累计奖励    
    
    reward_sum_his.append(reward_sum)
    
    df = pd.DataFrame(reward_sum_his, columns=['reward'])
    df.to_csv('./test.csv')
    
    for agent in agents:
        agent.brain.save_model()

def get_flie_path(args):
    r'''
    获取文件路径
    
    Input
    -----
    args:
    
    Output
    -----
    folder_path: dict[str, Any], 文件夹路径
        'data':  数据文件夹路径
        'image': 图像文件夹 
        'reward':历史累计奖励文件夹路径
        'loss':  历史损失值文件夹路径
        'model': 模型文件夹路径列表
    file_path: dict[str, Any], 文件路径
        'data':  数据文件路径, .npy
        'reward':历史累计奖励文件路径, .csv
        'loss':  历史损失值文件路径, .csv
        'model': 模型文件路径列表, list
    '''
    
    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    
    scene_name = env_configs['scene_name']                              # 场景名称
    
    # 文件夹路径设置---------------------------------------------------------------------
    scene_training_folder_path = f'./DRL_workspace/{scene_name}/'   # 场景模型训练工作区路径
    datas_folder_path = scene_training_folder_path + 'datas/'       # 数据保存文件夹路径
    results_folder_path = scene_training_folder_path + 'results/'   # 结果保存文件夹路径
    images_folder_path = results_folder_path + 'images/'            # 图像保存文件夹路径
    rewards_folder_path = results_folder_path + 'rewards/'          # 奖励保存文件夹路径
    losses_folder_path = results_folder_path + 'losses/'            # 损失值保存文件夹路径
    models_folder_path = results_folder_path + 'models/'            # 模型保存文件夹路径
    
    folder_path = {'data':datas_folder_path,'image':images_folder_path,
                   'reward':rewards_folder_path,'loss':losses_folder_path,
                   'model':models_folder_path}
    
    # 文件路径设置----------------------------------------------------------------------
    datas_file_path = datas_folder_path + str(args['configuration'])
    rewards_file_path = rewards_folder_path + file_name_str + '.csv'
    losses_file_path = losses_folder_path + file_name_str + '.csv'
    models_file_path = []
    for idx in range(len(target_configs)):
        model_file_path = models_folder_path + file_name_str + str(idx) + '.h5'
        models_file_path.append(model_file_path)  # 模型保存路径
    
    file_path = {'data':datas_file_path,'reward':rewards_file_path,
                 'loss':losses_file_path,'model':model_file_path}
    
    return folder_path, file_path
    
if __name__ == "__main__":
    np.set_printoptions(precision=3)                            # 设置打印精度为三位小数
    env_configs, target_configs = config_load('config-test')    # 读取配置文件
    args = set_args()                                           # 设置
    folder_path, file_path = get_flie_path(args)                # 获取文件路径
    env = Environment(folder_path, file_path, target_configs, 
                      args, **env_configs)                      # 初始化环境    
    env.run()                                                   # 运行