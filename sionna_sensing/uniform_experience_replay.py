"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import random
from collections import deque # 双端队列


class Memory(object):
    r'''
    记忆回放库
    '''
    def __init__(self, capacity):
        self.capacity = capacity                    # 记忆回放库的容量
        self.memory = deque(maxlen=self.capacity)   # 记忆回放库

    def remember(self, sample):
        self.memory.append(sample)                  # 将样本添加到记忆回放库中

    def sample(self, n):
        # 从记忆回放库中随机抽取n个样本，最大为记忆回放库的容量
        n = min(n, len(self.memory))  
        sample_batch = random.sample(self.memory, n)

        return sample_batch