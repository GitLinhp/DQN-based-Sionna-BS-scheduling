"""
Sionna 
包含部署DQN所需神经网络的代码
"""

import os
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, Concatenate
from keras.optimizers import *
import tensorflow as tf
from keras import backend as K

HUBER_LOSS_DELTA = 1.0

def huber_loss(y_true, y_predict):
    r'''
    Huber Loss 损失函数：增强平方误差损失函数对离群点的健壮性
        当预测误差小于 Delta 时，采用平方误差；
        当预测误差大于 Delta 时，采用线性误差。
    
    Input
    -----
    y_true: :class:`~tf.Tensor`
        实际值张量
    y_predict: :class:`~tf.Tensor`
        预测值张量
        
    Output
    -----
    K.mean(loss)
        损失函数平均值
    '''
    err = y_true - y_predict    
    
    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1) 

    return K.mean(loss)


class Brain(object):

    def __init__(self, state_size, action_size, brain_name, arguments):
        self.state_size = state_size                    # 状态空间
        self.action_size = action_size                  # 动作空间
        self.weight_backup = brain_name                 # 权重
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']
        self.test = arguments['test']
        self.num_nodes = arguments['number_nodes']
        self.dueling = arguments['dueling']
        self.optimizer_model = arguments['optimizer']
        self.model = self._build_model()
        self.model_ = self._build_model()

    def _build_model(self):

        if self.dueling:
            x = Input(shape=(self.state_size,))

            # 用于估计V(s)的一系列全连接层

            y11 = Dense(self.num_nodes, activation='relu')(x)
            y12 = Dense(self.num_nodes, activation='relu')(y11)
            y13 = Dense(1, activation="linear")(y12)

            # 用于估计A(s,a)的一系列全连接层

            y21 = Dense(self.num_nodes, activation='relu')(x)
            y22 = Dense(self.num_nodes, activation='relu')(y21)
            y23 = Dense(self.action_size, activation="linear")(y22)

            w = Concatenate(axis=-1)([y13, y23])

            # combine V(s) and A(s,a) to get Q(s,a)
            z = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                       output_shape=(self.action_size,))(w)
        else:
            x = Input(shape=(self.state_size,))

            # 用于估计Q(s,a)的一系列全连接层

            y1 = Dense(self.num_nodes, activation='relu')(x)
            y2 = Dense(self.num_nodes, activation='relu')(y1)
            z = Dense(self.action_size, activation="linear")(y2)

        model = Model(inputs=x, outputs=z)

        if self.optimizer_model == 'Adam':
            optimizer = Adam(lr=self.learning_rate, clipnorm=1.)
        elif self.optimizer_model == 'RMSProp':
            optimizer = RMSprop(lr=self.learning_rate, clipnorm=1.)
        else:
            print('Invalid optimizer!')

        model.compile(loss=huber_loss, optimizer=optimizer)
        
        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error:no file')
            else:
                model.load_weights(self.weight_backup)

        return model

    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  # x 为网络输入，y 为网络输出

        self.model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)

    def predict(self, state, target=False):
        if target:  # 从目标网络中获取预测 
            return self.model_.predict(state)
        else:       # 从本地网络中获取预测
            return self.model.predict(state)

    def predict_one_sample(self, state, target=False):
        return self.predict(state.reshape(1,self.state_size), target=target).flatten()

    def update_target_model(self):
        self.model_.set_weights(self.model.get_weights())

    def save_model(self):
        self.model.save(self.weight_backup)