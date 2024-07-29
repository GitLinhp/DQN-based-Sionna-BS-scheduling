"""
Sionna 
包含部署DQN所需神经网络的代码
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.losses import Huber

class Brain():
    r'''
    DQN 神经网络
    '''
    def __init__(self, state_space, action_space, brain_name, args):
        self.state_space = state_space
        self.action_space = action_space
        self.weight_backup = brain_name
        self.batch_size = args['batch_size']
        self.learning_rate = args['learning_rate']
        self.test = args['test']
        self.num_nodes = args['number_nodes']
        self.optimizer_model = args['optimizer']
        self.eval_net = self._build_model()         # 评估网络
        self.target_net = self._build_model()       # 目标网络

    def _build_model(self):
        x = keras.Input(shape=self.state_space)
                
        y1 = layers.Flatten()(x)
        y2 = layers.Dense(self.num_nodes, activation='relu')(y1)
        z = layers.Dense(self.action_space, activation='linear')(y2)
        model = keras.Model(inputs=x,outputs=z)
        
        # 配置优化器
        if self.optimizer_model == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_model == 'RMSProp':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            print('Invalid optimizer!')
        
        model.compile(loss=Huber(), optimizer=optimizer)
        
        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error:no file')
            else:
                model.load_weights(self.weight_backup)
        
        # model.summary()
        
        return model

    def train(self, inputs, outputs):  
        r'''
        模型训练
        
        Input
        -----
        inputs: :class:`~tf.tensor`
            网络输入    
        outputs: :class:`~tf.tensor`
            网络输出
        '''
        self.eval_net.fit(inputs, outputs, epochs=1, verbose=0)
        
    def predict(self, state, target=False):
        if target:  # 从目标网络中获取预测 
            return self.target_net.predict(state)
        else:       # 从评估网络中获取预测
            return self.eval_net.predict(state)
    
    def update_target_model(self):
        self.target_net.set_weights(self.eval_net.get_weights())
        
    def save_model(self):
        self.eval_net.save_weights(self.weight_backup)