import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from sionna.channel import cir_to_ofdm_channel
from sionna_sensing.dqn import DQN
from sionna_sensing.sensing_target import Target
from sionna_sensing.sensing_env import Environment
from sionna_sensing.config_load import config_load

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# 忽略来自 TensorFlow 的警告
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1) # 设置全局随机种子以保证可复现性

def store_data(fixed_route=False,is_los=False):
    r'''
    存储数据
    数据保存格式 target_num*[pos_now,crb(BS_num),pow_next]，target_num*(3+BS_num+3)
    
    Input
    -----
    fixed_route: bool
        是否固定轨迹
    '''
    if os.path.exists(data_save_path): # 如果文件存在，则加载文件
        datas = np.load(data_save_path)
    else:
        datas = np.empty((0,env.target_num,env.BS_num+2*3))
    # 采样次数
    if fixed_route:
        # 目标沿直线以恒定速率移动，每次移动的轨迹相同，只采集一次数据
        sample_num = 1
    else:
        sample_num = 3000
    for episode in range(sample_num): 
        print(f"====={episode}th episode start=====")
        t = time.time() # 用于计算数据采集用时
        env.reset()     # 重置环境
        inner_step = 0
        while True:
            if is_los:
                los = env._is_los() # 判断是否有 LoS
            data, done = env.get_data()
            print(f"\r【{episode}-{inner_step}th step】")
            for idx,target in enumerate(env.targets):
                if is_los:
                    print(f"\ttarget_{idx} pos:[{float(target.pos_now[0]):-6.1f},{float(target.pos_now[1]):-6.1f},{float(target.pos_now[2]):-6.2f}]\tv:{np.linalg.norm(target.velocity_now):.2f}\tlos:{los[idx]}")
                else:
                    print(f"\ttarget_{idx} pos:[{float(target.pos_now[0]):-6.1f},{float(target.pos_now[1]):-6.1f},{float(target.pos_now[2]):-6.2f}]\tv:{np.linalg.norm(target.velocity_now):.2f}")
            datas = np.concatenate((datas,data), axis=0)
            inner_step += 1
            if all(done): # 所有目标到达终点
                break
        np.save(data_save_path,datas) # 保存数据
    t = time.time()-t
    print(f"=====data collection finished=====")
    print(f'Total time cost:{t:.2f}s, Time cost per step:{t/(episode+1):.2f}s, Time cost per inner-step:{t/(episode+1)/inner_step:.2f}s')

def train(mode:str='offline'):
    r'''
    模型训练
    
    Input
    -----
    mode: str, default='offline'
        训练模式
            offline: 离线训练
            online: 在线训练
    '''
    # 初始化 DQN 网络
    RL = DQN(env.n_features,env.action_space,memory_size=1000,best_action=True, \
        best_prob=0.9,reward_decay=0.9,batch_size=256,replace_target_iter=5)
    if mode == 'offline':
        if os.path.exists(data_save_path) == False:
            store_data(fixed_route=False,is_los=False) # 采集数据
        offline_train(RL=RL,fixed_route=True) # 训练模型
    elif mode == 'online':
        online_train(RL=RL) # 训练模型

def offline_train(RL:DQN,fixed_route=False):
    r'''
    离线训练模型
    
    Input
    -----
    fixed_route: bool
        目标是否固定轨迹
    '''
    datas = np.load(data_save_path) # 加载数据
    data_size = len(datas)          # 数据大小
    data_counter = 0                # 数据计数
    step = 0
    for episode in range(3000):
        if fixed_route:
            np.random.shuffle(datas)          # 以第一维度随机打乱训练数据
            datas = datas[:RL.batch_size,:]   # 从打乱后的训练数据中取出 batch_size 条 
        correct_action = 0      # 记录每个episode的正确动作次数
        RL.reward_sum = 0       # 记录每个episode的累计奖励
        print(f"====={episode}th episode start=====")
        for inner_step in range(RL.batch_size):
            if fixed_route:
                index = inner_step
            else:
                index = (data_counter+inner_step)%data_size
                data_counter += 1
            data = datas[index,:]
            pos_now = data[:,:3]
            pos_next = data[:,-3:]
            crbs = data[:,3:-3]
            observation = np.reshape(pos_now[:,:2]/[10.,400.], (-1))  # 将位置归一化
            # 只传入目标位置x和y坐标进行训练
            print(f'\r【{step}-{inner_step}th step】')
            for idx in range(env.target_num):
                action, action_type = RL.choose_action(observation, crbs)  # 选择动作
                reward = env._get_reward(action,method='crb',crb_target=crbs)[idx,action] # 获取奖励
                RL.reward_sum += reward # 累计奖励
                if reward == 1.:
                    correct_action += 1 # 统计正确动作次数
                print(f"\tpos:[{float(pos_now[idx,0]):.1f},{float(pos_now[idx,1]):.1f},{float(pos_now[idx,2]):.2f}]\tBS:{action}({action_type})\treward:{reward:.4f}")
            observation_ = np.reshape(pos_next[:,:2]/[10.,400.], (-1))
            RL.store_transition(observation, action, reward, observation_) # 存储状态转移
        step += 1
        RL.reward_sum_his = np.append(RL.reward_sum_his, RL.reward_sum) # 历史累积奖励
        correct_action_rate = correct_action/RL.batch_size              # 正确动作率
        RL.correct_action_rate_his = np.append(RL.correct_action_rate_his, correct_action_rate) # 历史正确动作率
        print(f"correct action rate:{correct_action_rate*100:.2f}%")
        RL.learn(model_save_path)

def online_train(RL:DQN,show_scene_image=False):
    step = 0
    for episode in range(3010): 
        print(f"====={episode}th episode start=====")
        env.reset()                         # 重置环境
        observation = env.get_observation() # 获取观测值
        correct_action = 0      # 记录每个episode的正确动作次数
        RL.reward_sum = 0       # 记录每个episode的累计奖励
        inner_step = 0
        while True:
            env.paths = env.scene.compute_paths(**env.ray_tracing_params)   # 计算路径
            crbs = env._get_crbs()                                          # 获取 CRB
            # 获取场景图片
            if show_scene_image:
                env._get_scene_image(filename="scene_test.png") 
            action, action_type = RL.choose_action(observation, crbs)       # 选择动作
            observation_, reward, done = env.step(action, crbs)             # 执行动作
            RL.reward_sum += reward # 累计奖励
            if reward == 1.:
                correct_action += 1 # 统计正确次数
            print(f"\r【{step}-{inner_step}th step】")
            for idx,target in enumerate(env.targets):
                print(f"\ttarget_{idx} pos:[{float(target.pos_now[0]):.1f},{float(target.pos_now[1]):.1f},{float(target.pos_now[2]):.2f}]\tBS:{action}({action_type})\treward:{reward:.4f}")
            
            RL.store_transition(observation, action, reward, observation_) # 存储到记忆回放库中
            observation = observation_
            inner_step += 1
            if all(done): # 所有目标到达终点
                step += 1
                if step >= 10:
                    RL.reward_sum_his = np.append(RL.reward_sum_his, RL.reward_sum)
                    correct_action_rate = correct_action/(inner_step+1)
                    RL.correct_action_rate_his = np.append(RL.correct_action_rate_his, correct_action_rate)
                    print(f"correct action rate:{correct_action_rate*100:.2f}%")    # 正确动作率
                    RL.learn(model_save_path)  # 模型学习
                break

def motion_test():
    for episode in range(3000):
        print(f"====={episode}th episode start=====")
        env.reset()     # 重置环境
        inner_step = 0
        while True:
            print(f"\r【{episode}-{inner_step}th step】")
            for idx,target in enumerate(env.targets):
                _,_,env.done[idx] = target.move(env.DASX,env.DASY,env.TIME_SLOT)
                print(f"\ttarget_{idx} pos:[{float(target.pos_now[0]):.1f},{float(target.pos_now[1]):.1f},{float(target.pos_now[2]):.2f}]\tv:{np.linalg.norm(target.velocity_now):.2f}")
            inner_step += 1
            if all(env.done): # 所有目标到达终点
                break

if __name__ == "__main__":
    np.set_printoptions(precision=1)                                # 设置打印精度为一位小数
    config_name = 'config copy'                                     # 配置文件名
    data_name = 'datas-0709-single-target'                          # 数据文件名
    # data_name = 'datas-0707'                                      # 数据文件名
    model_name = 'dqn-0709-single-target-offline'                           # 模型文件名
    
    config_path = f'./sionna_sensing/configs/{config_name}.json'    # 配置文件路径
    env_configs, target_configs = config_load(config_path)          # 读取配置文件
    scene_name = env_configs['scene_name']                          # 场景名称
    scene_training_path = f'./DL_workspace/{scene_name}'            # 场景模型训练工作区路径
    data_save_path = f'{scene_training_path}/datas/{data_name}.npy' # 数据保存路径
    image_save_path = f'{scene_training_path}/images/'              # 场景图像保存路径
    model_save_path = f'{scene_training_path}/models/{model_name}/' # 模型保存路径
    env = Environment(image_save_path,target_configs,**env_configs) # 初始化环境
    # train(mode='offline')                                            # 训练模型
    motion_test()