r"""
Sionna 
包含部署DQN网络和其扩展（Double DQN，Dueling DQN，优先经验回放的DQN）的代码
"""

import numpy as np
import random

from sionna_sensing.multi_targets_sensing.brain import Brain                     # 神经网络
from uniform_experience_replay import UER   # 统一经验回放

# Epsilon-greedy 策略
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

# 
MIN_BETA = 0.4
MAX_BETA = 1.0

class Agent(object):
    
    epsilon = MAX_EPSILON
    beta = MIN_BETA

    def __init__(self, state_space, action_space, bee_index, brain_name, arguments):
        self.state_space = state_space      # 状态空间的大小
        self.action_space = action_space    # 动作空间的大小
        self.bee_index = bee_index      
        self.learning_rate = arguments['learning_rate'] # 学习率
        self.gamma = 0.95                               # 折扣因子
        self.brain = Brain(self.state_space, self.action_space, brain_name, arguments) # 神经网络

        # 初始化记忆回放库
        self.memory = UER(arguments['memory_capacity'])

        self.update_target_frequency = arguments['target_frequency']    # 更新目标网络的频率
        self.batch_size = arguments['batch_size']                       # 批次大小
        self.step = 0                                                   # 记录步数
        self.test = arguments['test']                                   # 测试模型训练效果
        if self.test:
            self.epsilon = MIN_EPSILON

    def greedy_actor(self, state):
        r'''
        epsilon-greedy 动作策略
        
        Input
        ----
        state: np.array
            当前状态
            
        Output
        -----
        action: int
            动作
        action_type: str
            动作类型
                'G': greedy
                'R': random
        '''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space), 'R'
        else:
            return np.argmax(self.brain.predict_one_sample(state)), 'G'

    def select_sensing_target_uer(self, batch):
        r'''
        选择感知目标，采用统一经验回放
        
        Input
        -----
        batch: 一批次数据
        '''
        batch_len = len(batch)
        
        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_space))
        y = np.zeros((batch_len, self.action_space))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1][self.bee_index]
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                t[a] = r + self.gamma * np.amax(pTarget_[i])

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y]

    def observe(self, sample):
        r'''
        获取观测
        '''
        self.memory.remember(sample)

    def decay_epsilon(self):
        r'''
        基于经验缓慢降低 Epsilon
        '''
        
        self.step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
            self.beta = MAX_BETA
        else:
            if self.step < self.max_exploration_step:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.step)/self.max_exploration_step
                self.beta = MAX_BETA + (MIN_BETA - MAX_BETA) * (self.max_exploration_step - self.step)/self.max_exploration_step
            else:
                self.epsilon = MIN_EPSILON

    def replay(self):
        r'''
        经验回放
        '''
        batch = self.memory.sample(self.batch_size)
        x, y = self.find_targets_uer(batch)
        self.brain.train(x, y)

    def update_target_model(self):
        r'''
        更新目标网络模型
        '''
        if self.step % self.update_target_frequency == 0:
            self.brain.update_target_model()