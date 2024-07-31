"""
优先经验回放
"""

import random
from sionna_sensing.multi_targets_sensing.sum_tree import SumTree as ST


class PER():
    r'''
    优先经验回放，Prioritized Experience Replay
    '''
    e = 0.05

    def __init__(self, capacity, pr_scale):
        self.capacity = capacity        # 经验回放库的容量
        self.memory = ST(self.capacity) # 经验回放库
        self.pr_scale = pr_scale
        self.max_pr = 0

    def get_priority(self, error):
        return (error + self.e) ** self.pr_scale

    def remember(self, sample, error):
        p = self.get_priority(error)

        self_max = max(self.max_pr, p)
        self.memory.add(self_max, sample)

    def sample(self, n):
        sample_batch = []
        sample_batch_indices = []
        sample_batch_priorities = []
        num_segments = self.memory.total() / n

        for i in range(n):
            left = num_segments * i
            right = num_segments * (i + 1)

            s = random.uniform(left, right)
            idx, pr, data = self.memory.get(s)
            sample_batch.append((idx, data))
            sample_batch_indices.append(idx)
            sample_batch_priorities.append(pr)

        return [sample_batch, sample_batch_indices, sample_batch_priorities]

    def update(self, batch_indices, errors):
        for i in range(len(batch_indices)):
            p = self.get_priority(errors[i])
            self.memory.update(batch_indices[i], p)