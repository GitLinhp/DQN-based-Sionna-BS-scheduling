"""
统一经验回放
"""

import random
from collections import deque # 双端队列


class UER(object):
    r'''
    经验回放库，uniform experience replay
    '''
    def __init__(self, capacity):
        self.capacity = capacity                    # 经验回放库的容量
        self.memory = deque(maxlen=self.capacity)   # 经验回放库的内存

    def remember(self, sample):
        r'''
        将样本添加到经验回放库中
        
        Input
        -----
        sample: 
            样本
        '''
        self.memory.append(sample)                  

    def sample(self, n):
        r'''
        从记忆回放库中随机抽取n个样本，最大为记忆回放库的容量
        
        Input
        -----
        n: int
            抽取的样本数
        '''
        n = min(n, len(self.memory))  
        sample_batch = random.sample(self.memory, n)

        return sample_batch