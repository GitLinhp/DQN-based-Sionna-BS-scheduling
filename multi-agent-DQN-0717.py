import os
gpu_num = 1 # 制定使用的 GPU 序号，使用 "" 来启用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
from sionna_sensing.multi_targets_sensing.dqn_agent_test import Agent
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
ARG_LIST = ['learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency', 'maximum_exploration',
            'max_timestep', 'first_step_memory', 'replay_steps', 'number_nodes', 'target_type', 'memory',
            'prioritization_scale', 'dueling', 'agents_number', 'reward_mode']

def get_name_brain(args, idx):
    # 获取神经网络文件名称
    # file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    
    return models_save_path+f"{idx}.h5"

def get_name_rewards(args):
    # 获取奖励函数文件名称
    # file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    # return './results_agents_landmarks/rewards_files/' + file_name_str + '.csv'
    return results_save_path+f"/rewards/"

def get_name_timesteps(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results_agents_landmarks/timesteps_files/' + file_name_str + '.csv'

def set_DQN_args():
    r'''
    设置 DQN 参数，可通过命令行更改
    '''
    # 参数
    parser = argparse.ArgumentParser()
    # DQN 参数
    parser.add_argument('-e', '--episode-number', default=3000, type=int, help='训练代数，Number of episodes')
    parser.add_argument('-l', '--learning-rate', default=0.9, type=float, help='学习率，Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='Adam',
                            help='优化方法，Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=1000, type=int, help='内存容量，Memory capacity')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='批次大小，Batch size')
    parser.add_argument('-t', '--target-frequency', default=100, type=int,
                        help='目标网络更新步数间隔，Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=3000, type=int, help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory', default=0, type=float,
                        help='填满内存的初始步骤数，Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=4, type=float, help='更新评估网络的步数间隔，steps between updating the network')
    parser.add_argument('-nn', '--number-nodes', default=512, type=int, help='神经网络每层的节点数，Number of nodes in each layer of NN')

    parser.add_argument('-gn', '--gpu-num', default='3', type=str, help='训练使用的 GPU 编号')
    parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase if "store_false"')
    # 参数待完善
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

    
if __name__ == "__main__":
    np.set_printoptions(precision=3)                                # 设置打印精度为一位小数
    config_name = 'config test'                                     # 配置文件名
    data_name = 'datas-0726'                                        # 数据文件名
    model_name = 'dqn-0726'                                         # 模型文件名
    
    config_path = f'./sionna_sensing/config/{config_name}.json'     # 配置文件路径
    env_configs, target_configs = config_load(config_path)          # 读取配置文件
    scene_name = env_configs['scene_name']                          # 场景名称
    scene_training_path = f'./DRL_workspace/{scene_name}'           # 场景模型训练工作区路径
    datas_save_path = f'{scene_training_path}/datas/'               # 数据保存路径
    
    results_save_path = f'{scene_training_path}/results/'           # 模型保存路径
    images_save_path = results_save_path+'images/'                 # 场景图像保存路径
    rewards_save_path = results_save_path+'rewards/'               # 奖励保存路径
    models_save_path = results_save_path+'models/'
    
    if os.path.exists(models_save_path) == False:
        os.makedirs(models_save_path)
    
    args = set_DQN_args()
    
    env = Environment(images_save_path,target_configs,args,**env_configs) # 初始化环境    
        
    agents = [] # 初始化智能体
    for b_idx in range(env.agent_num):
        # brain_file: ./DRL_workspace/one_way_street/results/models/0.h5
        brain_file = get_name_brain(args, b_idx)
        agent = Agent(env.state_space,env.action_space,b_idx,memory_size=1000,\
            reward_decay=0.9,batch_size=256,brain_name=brain_file,args=args)
        agents.append(agent)
    
    env.run(agents,rewards_save_path)