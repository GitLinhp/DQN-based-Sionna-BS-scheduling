"""
Sionna 
包含部署DQN所需神经网络的代码
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib as plt
import random
from matplotlib import ticker
from sionna_sensing.multi_targets_sensing.brain import Brain
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import backend as K
from sionna_sensing.multi_targets_sensing.uniform_experience_replay import UER   # 统一经验回放

# Epsilon-greedy 策略
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

# 
MIN_BETA = 0.4
MAX_BETA = 1.0

class Agent():
    r'''
    DQN 智能体
    '''
    def __init__(self, state_space, action_space, brain_idx,
                 reward_decay=0.9, brain_name=None, args=None):
        self.state_space = state_space      # 状态空间
        self.action_space = action_space    # 动作空间
        self.learning_rate = args['learning_rate']  # 学习率
        self.gamma = reward_decay           # 折损因子
        
        self.b_idx = brain_idx
        self.memory = UER(args['memory_capacity'])  # 初始化记忆回放库
        
        # self.best_action = best_action      # 是否使用最优动作
        # self.best_prob = best_prob          # 最优动作概率
        
        if brain_name is None:
            raise('Brain name is None!')
        # 神经网络
        self.brain = Brain(self.state_space, self.action_space, brain_name, args)
        
        self.batch_size = args['batch_size']        # 批次大小
        self.update_target_frequency = args['target_frequency']
        self.max_exploration_step = args['maximum_exploration']
        self.learning_step = 0                 # 学习步数
        self.test = args['test']
        if self.test:
            self.epsilon = MIN_EPSILON
            self.beta = MAX_BETA
        else:
            self.epsilon = MAX_EPSILON
            self.beta = MIN_BETA
    
    def choose_action(self, state):
        r'''
        epsilon-greedy 动作策略
        
        Input
        ----
        state: np.array, 当前状态
            
        Output
        -----
        action: int, 动作
        action_type: str, 动作类型
            'G': greedy
            'R': random
        '''
        
        state = np.expand_dims(state, axis=0)
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_space)
            action_type = 'R'
        else:
            action_value = self.brain.eval_net(state)
            action = np.argmax(action_value)
            action_type = 'G'
        
        # 未选择目标
        if action == self.action_space-1:
            action = -1
        
        return action, action_type
        
    def observe(self, sample):
        r'''
        获取观测
        '''
        self.memory.remember(sample)
    
    def decay_epsilon(self):
        r'''
        基于经验缓慢降低 Epsilon
        '''
        
        self.learning_step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
            self.beta = MAX_BETA
        else:
            if self.learning_step < self.max_exploration_step:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.learning_step)/self.max_exploration_step
                self.beta = MAX_BETA + (MIN_BETA - MAX_BETA) * (self.max_exploration_step - self.learning_step)/self.max_exploration_step
            else:
                self.epsilon = MIN_EPSILON
    
    def select_sensing_targets_uer(self, batch):
        r'''
        寻找目标，采用统一经验回放
        
        Input
        -----
        batch: :class:`collections.deque`, DQN 经验回放库
            
        Output
        -----
        inputs: :class:`np.array`, 神经网络输入
        outputs: :class:`np.array`, 神经网络输出
        '''
        
        # batch:(state, actions, reward, state_, done)
        batch_len = len(batch)
        
        # [batch_len, target_num, 2]
        states = np.array([o[0] for o in batch])
        # [batch_len, target_num, 2]
        states_ = np.array([o[3] for o in batch])
        
        # verbose: 0: 不显示进度, 1: defalt, 显示进度条
        q_eval = self.brain.eval_net.predict(states, verbose=0)
        q_next = self.brain.target_net.predict(states_, verbose=0)
        q_target = q_eval.copy()
        
        for idx in range(batch_len):
            action = batch[idx][1]
            reward = batch[idx][2]
            
            q_target[idx, action] = reward + self.gamma * tf.reduce_max(q_next[idx])
        
        return states, q_target
    
    def replay(self):
        r'''
        经验回放
        '''
        
        batch = self.memory.sample(self.batch_size)
        inputs, outputs = self.select_sensing_targets_uer(batch)
        self.brain.train(inputs, outputs)
    
    def update_target_model(self):
        r'''
        更新目标模型
        '''
        
        if self.learning_step % self.update_target_frequency == 0:
            self.brain.update_target_model()
