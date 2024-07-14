import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

IMAGE_RESOLUTION = [600,180] # 场景图像分辨率
class DQNnet(tf.keras.Model):
    r'''
    DQN网络
    '''
    def __init__(self, num_action,input_shape=(IMAGE_RESOLUTION[0],IMAGE_RESOLUTION[1],3),trainable=True):
        super().__init__('mlp_q_network')
        if trainable:
            self.resNet = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=input_shape)
            self.resNet.trainable = True
            self.flatten = tf.keras.layers.Flatten()
            self.dense = tf.keras.layers.Dense(512,activation='relu',trainable=True)
            self.out = tf.keras.layers.Dense(num_action,trainable=True)
            
        else:
            self.resNet = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=input_shape)
            self.resNet.trainable = False
            self.flatten = tf.keras.layers.Flatten()
            self.dense = tf.keras.layers.Dense(512,activation='relu',trainable=False)
            self.out = tf.keras.layers.Dense(num_action,trainable=False)
    
    def call(self, inputs):
        # inputs: [batch_size, num_features]
        inputs = tf.convert_to_tensor(inputs)
        x = self.dense(inputs)
        return self.out(x)

class DQN():
    def __init__(self, num_feature, num_action, learning_rate=0.01, reward_decay=0.9, e_greedy=0.2,replace_target_iter=100, memory_size=1000, batch_size=32,path=None,best_action=False,best_prob=0.6):
        self.num_feature = num_feature      # 特征数量
        self.num_action = num_action        # 动作数量
        self.lr = learning_rate             # 学习率
        self.gamma = reward_decay           # 奖励衰减
        self.epsilon = e_greedy             # epsilon-贪心策略
        self.epsilon_increment = 0.001      # 贪心策略增量
        self.epsilon_max = 0.9              # 贪心策略最大值
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size      # 记忆库大小
        self.batch_size = batch_size        # 批次大小
        self.memory = np.zeros((self.memory_size, num_feature*2+2)) # feature + action + reward + feature_ 
        self.memory_counter = 0             # 记忆库计数
        self.learn_step_counter = 0         # 学习步数
        self.loss_his = []                  # 历史损失函数
        self.mean_cost = 99999              # 平均开销
        self.best_action = best_action      # 是否使用最优动作
        self.best_prob = best_prob          # 最优动作概率
        self.reward_sum = 0                 # 累积奖励
        self.reward_sum_his = []            # 历史累积奖励
        self.correct_action_rate_his = []   # 历史正确动作率
        
        self.eval_net = DQNnet(self.num_action)         # 评估网络
        self.target_net = DQNnet(self.num_action,False) # 目标网络
        self.eval_net.compile(optimizer=tf.keras.optimizers.Adam(self.lr),loss=tf.keras.losses.Huber())
        if path is not None and isinstance(path,str) and os.path.exists(path):
            self.eval_net = tf.keras.models.load_model(path)
            self.target_net = tf.keras.models.load_model(path.replace('eval','target'))
    
    def reward_nomalization(reward):
        r'''
        奖励归一化
        
        Input
        -----
        reward: np.array
            奖励
        '''
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        normailzed_reward = (reward-mean_reward)/std_reward
        return normailzed_reward
           
    def store_transition(self, s, a, r, s_):
        r'''
        储存transition，用于经验回放
        
        Input
        -----
        s: np.array
            当前状态
        a: int
            动作
        r: float
            奖励
        s_: np.array
            下一状态
        '''
        transition = np.hstack((s, [a, r], s_)) # 在水平方向上平铺
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def choose_action(self, observation:np.array, crbs=None):
        r'''
        选择动作
        
        Input
        -----
        observation: np.array
            观测值
        crbs: np.array
            每个动作的crb值
            
        Output
        -----
        action: int
            动作
        action_type: str
            动作类型
                'M': model
                'R': random
                'B': best
        '''
        if crbs is None:
            raise ValueError('crbs is None')
        observation = np.expand_dims(observation, axis=0)
        action_type='R'
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(observation)
            action = np.argmax(actions_value)
            action_type='M'
        else:
            best_action = np.random.randint(0, self.num_action)
            action_type='R'
            if self.best_action and np.random.rand() < self.best_prob:
                action_type='B'
                best_action = np.argmin(crbs)
            action = best_action
        return action, action_type

    def _replace_target_params(self):
        self.target_net.set_weights(self.eval_net.get_weights())
    
    def learn(self,model_save_path):
        r'''
        模型学习
        
        Input
        -----
        model_save_path: str
            模型保存路径
        '''
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_memory_as_qnext_input = batch_memory[:, -self.num_feature:]
        batch_memory_as_qeval_input = batch_memory[:, :self.num_feature]
        q_next = self.target_net.predict(batch_memory_as_qnext_input)
        q_eval = self.eval_net.predict(batch_memory_as_qeval_input)
        
        q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        action = batch_memory[:, self.num_feature].astype(int)
        reward = batch_memory[:, self.num_feature+1]
        q_target[batch_index, action] = reward + self.gamma * tf.reduce_max(q_next, axis=1)
        
        # 训练
        self.loss = self.eval_net.train_on_batch(batch_memory_as_qeval_input, q_target)
        print(f"loss: {self.loss}")
        self.loss_his.append(np.mean(self.loss))
        
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            new_mean_cost = np.mean(self.loss_his)
            print('\ntarget_params_replaced\n')
            # 保存模型
            if new_mean_cost < self.mean_cost:
                self.mean_cost = new_mean_cost
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            self.save_model(model_save_path)
            print(f"model saved, mean cost: {self.mean_cost}")
        self.learn_step_counter += 1
        
    def save_model(self,path):
        r'''
        保存模型，需要修改
        
        Input
        -----
        path: str 
            模型保存路径
        '''
        #time: Month-Day-Hour-Minute
        self.eval_net.save_weights(f"{path}/eval.h5")
        self.target_net.save_weights(f"{path}/target.h5")
        # 保存损失函数
        plt.figure()
        plt.plot(np.arange(len(self.loss_his)), self.loss_his)
        plt.ylabel('loss')
        plt.xlabel('training steps')
        plt.title('Loss by Training Steps')
        plt.savefig(f'{path}/loss.png')
        np.savetxt(f'{path}/loss.txt',self.loss_his)
        # 保存累积奖励
        plt.figure()
        plt.plot(np.arange(len(self.reward_sum_his)), self.reward_sum_his)
        plt.ylabel('reward sum')
        plt.xlabel('training steps')
        plt.title('Reward Sum by Training Steps')
        plt.savefig(f'{path}/reward sum.png')
        np.savetxt(f'{path}/reward sum.txt',self.reward_sum_his)
        # 保存正确动作率
        plt.figure()
        plt.ylim(0.6, 1.0)
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        plt.plot(np.arange(len(self.correct_action_rate_his)), self.correct_action_rate_his)
        plt.ylabel('correct action rate')
        plt.xlabel('training steps')
        plt.title('Correct Action Rate by Training Steps')
        plt.savefig(f'{path}/correct action rate.png')
        np.savetxt(f'{path}/correct action rate.txt',self.correct_action_rate_his) 
        # 保存平滑后的正确动作率
        print(f"Everage correct action rate of last 100 episodes:{np.mean(self.correct_action_rate_his[-100:])*100:.2f}%")
        print(f"Best correct action:{np.max(self.correct_action_rate_his)*100:.2f}%")
        interval = 30
        average_correct_action_rate = []
        for i in range(int(3000/interval)):
            average_correct_action_rate = np.append(average_correct_action_rate ,np.mean(self.correct_action_rate_his[i*interval:(i+1)*interval]))
        plt.figure()
        plt.ylim(0.6, 1.0)
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        plt.plot(np.arange(0,3000,interval), average_correct_action_rate)
        plt.ylabel('correct action rate')
        plt.xlabel('training steps')
        plt.title(f'Average Correct Action Rate per {interval} episodes by Training Steps')
        if path is None:
            raise ValueError('path is None')
        plt.savefig(f'{path}/smooth correct action rate.png')    