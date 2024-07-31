"""
Sionna 
包含部署DQN所需神经网络的代码
"""

import os
from tensorflow import keras
from keras import layers
from keras.losses import Huber

class Brain():
    r'''
    DQN 神经网络
    '''
    def __init__(self, state_space, action_space, brain_name, args):
        self.state_space = state_space              # 状态空间
        self.action_space = action_space            # 动作空间
        self.weight_backup = brain_name             # 模型权重文件路径
        self.batch_size = args['batch_size']        # 训练批次大小
        self.learning_rate = args['learning_rate']  # 学习率
        self.test = args['test']                    # 测试模型训练效果
        self.num_nodes = args['number_nodes']       # 网络节点数
        self.optimizer_model = args['optimizer']    # 优化器类型
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
        
        # 模型编译
        model.compile(loss=Huber(), optimizer=optimizer)
        
        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error:no file')
            else:
                model.load_weights(self.weight_backup)

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
        r'''
        模型预测
        
        Input
        -----
        state: :class:`~tf.tensor`
            状态
        target: bool
            是否为目标网络
        
        Output
        -----
        prediction: :class:`~tf.tensor`
            预测结果
        '''
        if target:  # 从目标网络中获取预测 
            return self.target_net.predict(state)
        else:       # 从评估网络中获取预测
            return self.eval_net.predict(state)
    
    def update_target_model(self):
        self.target_net.set_weights(self.eval_net.get_weights())
        
    def save_model(self):
        self.eval_net.save_weights(self.weight_backup)