import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
from mysionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from mysionna.rt.scattering_pattern import *
from mysionna.rt.scene import Target, load_sensing_scene
import PIL.Image as Image

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# 忽略来自 TensorFlow 的警告
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1) # 设置全局随机种子以保证可复现性

model_save_path = './models/cross_road/' # 模型保存路径
image_save_path = './images/cross_road/' # 场景图像保存路径
DAS = 200 # Default Area Size,默认目标活动正方形区域范围（200m）
# 目标限制移动范围为 DASX x DASY 的矩形区域
# 为目标可移动范围, 并非场景大小
DASX = 40 # Default Area Size X,默认目标活动区域x轴范围（200m）
DASY = 800 # Default Area Size Y,默认目标活动区域y轴范围（800m）
VCRT = 0.05 # Velocity Change Rate,速度变化概率 m/s
VCS = 5 # Velocity Change Size,速度变化大小，即一次最多变化多少 m/s
VCRG = [10,20] # Velocity Change Range,速度变化范围 m/s (0.28m/s~~1km/h)
TIME_SLOT = 0.05 # 时间间隔 s
IMAGE_RESOLUTION = [600,180] # 场景图像分辨率
# 目标移动策略 random,graph
# random: 在区域内随机移动，需更改配置DAS，目标将在以原点为中心，边长为2*DAS的正方形区域内随机移动
# graph: 按照指定路线移动,主要用于模拟车辆轨迹/路上行人轨迹。需构建环境时提供如下参数：
#        start_points: [num1,3] float,指定的起点
#        points: [num2,3] float,指定的移动点
#        end_points: [num3,3] float,指定的终点
#        point_bias: float,移动点偏移范围,目标会把目的点设置为以point为中心，point_bias为半径的圆内的随机点
#        point_path:[num1+num2+num3,num1+num2+num3] int,邻接矩阵，表示点之间的路径关系（有向图表示）,前num1个为start_points,中间num2个为points,最后num3个为end_points
#        DAS:正方形限制移动范围边长
#        DASX:x轴限制移动范围
#        DASY:y轴限制移动范围
MOVE_STRATEGY = 'graph' 

class DQNnet(tf.keras.Model):
    # 以二维数据作为输入
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
        self.num_feature = num_feature
        self.num_action = num_action
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_increment = 0.001
        self.epsilon_max = 0.9
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size, num_feature*2+2)) # feature + action + reward + feature_ 
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.loss_his = []
        self.mean_cost = 99999
        self.best_action = best_action
        self.best_prob = best_prob
        self.reward_sum = 0
        self.reward_sum_his = []
        self.correct_action_rate_his = [] 
        
        self.eval_net = DQNnet(self.num_action) # 评估网络
        self.target_net = DQNnet(self.num_action,False) # 目标网络
        self.eval_net.compile(optimizer=tf.keras.optimizers.Adam(self.lr),loss=tf.keras.losses.Huber())
        # self.loss_func = tf.losses.mean_squared_error
        # self.optimizer = tf.keras.optimizers.Adam(self.lr)
        if path is not None and isinstance(path,str) and os.path.exists(path):
            self.eval_net = tf.keras.models.load_model(path)
            self.target_net = tf.keras.models.load_model(path.replace('eval','target'))
    
    def reward_nomalization(reward):
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        normailzed_reward = (reward-mean_reward)/std_reward
        return normailzed_reward
           
    def store_transition(self, s, a, r, s_):
        # 储存transition，用于经验回放
        transition = np.hstack((s, [a, r], s_)) #在水平方向上平铺
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def choose_action(self, observation, crbs=None):
        if crbs is None:
            raise ValueError('crbs is None')
        action_type='R'
        observation = np.reshape(observation, (1,-1))
        # print(observation.shape)
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(observation)
            action = np.argmax(actions_value)
            action_type='M'
        else:
            best_action = np.random.randint(0, self.num_action)
            action_type='R'
            if self.best_action and np.random.rand() < self.best_prob:
                action_type='B'
                # best_reward = -1
                # for action in range(self.num_action):
                #     if env.los[action]:
                #         reward = env._get_reward(action)
                #         if reward > best_reward:
                #             best_reward = reward
                #             best_action = action
                best_action = np.argmin(crbs)
            action = best_action
        return action, action_type

    def _replace_target_params(self):
        self.target_net.set_weights(self.eval_net.get_weights())
    
    def learn(self):
        '''学习
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
        
        # train
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
        #time: Month-Day-Hour-Minute
        #precent = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
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
        
class Environment():
    def __init__(self,**kwargs):
        self.scene = None
        # 环境路径
        self.env_path = kwargs.get('env_path','./scenes/One_Way_Street/one_way_street.xml')
        # 基站参数---------------------------------------------------------
        # 基站个数
        self.action_space = kwargs.get('action_space',7)
        self.BS_num = self.action_space 
        # 基站位置，要与个数对应, 基站位置过高可能导致感知效果不佳
        self.BS_pos = kwargs.get('BS_pos',np.array([[-24,-360,10],[21,-240,10],[-24,-120,10],[26,0,10],[-30,120,10],[27,240,10],[-30,360,10]]))
        # 目标移动范围参数---------------------------------------------------------
        if MOVE_STRATEGY == 'graph':
            self.start = True
            # end_points: [num1,3] float,指定的起点
            self.start_points = kwargs.get('start_points',np.array([[0,-400,0.05]]))
            # points: [num2,3] float,指定的移动点
            self.points = kwargs.get('points',np.array([[0,0,0.05]]))
            # end_points: [num3,3] float,指定的终点
            self.end_points = kwargs.get('end_points',np.array([[0,400,0.05]]))
            # range
            points_list = np.concatenate((self.start_points,self.points,self.end_points),axis=0)
            self.x_range = np.max(np.max(np.abs(points_list[:,0])))
            self.y_range = np.max(np.max(np.abs(points_list[:,1])))
            # point_bias: float,移动点偏移范围,目标会把目的点设置为以point为中心，point_bias为半径的圆内的随机点
            self.point_bias = kwargs.get('point_bias',0)
            # point_path:[num1+num2+num3,num1+num2+num3] int,邻接矩阵，表示点之间的路径关系（有向图表示）,前num1个为start_points,中间num2个为points,最后num3个为end_points
            self.point_path = kwargs.get('point_path',np.array([[0,1,0],[0,0,1],[0,0,0]]))
            # 起点，移动点，终点数
            self.num_start_points = len(self.start_points)
            self.num_points = len(self.points)
            self.num_end_points = len(self.end_points)
            # print(f"start_points:{self.num_start_points},points:{self.num_points},end_points:{self.num_end_points}")
            num_path = len(self.point_path)
            if num_path != self.num_start_points + self.num_points + self.num_end_points:
                raise ValueError('point_path must be a (num_start_points+num_points+num_end_points) x (num_start_points+num_points+num_end_points) matrix')
        # 目标参数---------------------------------------------------------
        self.target_num = kwargs.get('target_num',1)
        self.target_name = kwargs.get('target_name','car')
        self.target_path = kwargs.get('target_path','meshes/car.ply')
        self.target_material = kwargs.get('target_material','itu_concrete')
        self.target_size = kwargs.get('target_size',1.3) # 目标的尺寸，用于在计算估计误差时减去的偏移量，即偏移量在目标尺寸范围内视为0
        # 天线配置参数 ---------------------------------------------------------
        self.tx_params = {
            "num_rows": kwargs.get('num_tx_rows',1),
            "num_cols": kwargs.get('num_tx_cols',1),
            "vertical_spacing": kwargs.get('vertical_spacing',0.5),
            "horizontal_spacing": kwargs.get('horizontal_spacing',0.5),
            "pattern": kwargs.get('pattern','dipole'),
            "polarization": kwargs.get('polarization','V'),
            "polarization_model": kwargs.get('polarization',2)
        }
        self.rx_params = {
            "num_rows": kwargs.get('num_rx_rows',1),
            "num_cols": kwargs.get('num_rx_cols',1),
            "vertical_spacing": kwargs.get('vertical_spacing',0.5),
            "horizontal_spacing": kwargs.get('horizontal_spacing',0.5),
            "pattern": kwargs.get('pattern','dipole'),
            "polarization": kwargs.get('polarization','V'),
            "polarization_model": kwargs.get('polarization',2)
        }
        self.frequency = kwargs.get('frequency',6e9)
        self.synthetic_array = kwargs.get('synthetic_array',True)
        self.BS_pos_trainable = kwargs.get('BS_pos_trainable',False)
        # 光线追踪参数 ---------------------------------------------------------
        self.ray_tracing_params = {
            "max_depth": kwargs.get('max_depth',1),
            "method": kwargs.get('method','fibonacci'),
            "num_samples": kwargs.get('num_samples',int(4e5 * self.BS_num)),
            "los": kwargs.get('los',True),
            "reflection": kwargs.get('reflection',True),
            "diffraction": kwargs.get('diffraction',True),
            "scattering": kwargs.get('scattering',True),
            "scat_keep_prob": kwargs.get('scat_keep_prob',0.01),
            "edge_diffraction": kwargs.get('edge_diffraction',True),
            "check_scene": kwargs.get('check_scene',True),
            "scat_random_phases": kwargs.get('scat_random_phases',False)
        }
        self.scat_keep_prob_fixed = self.ray_tracing_params["scat_keep_prob"]
        # 频域信道参数 ---------------------------------------------------------
        self.subcarrier_spacing = kwargs.get('subcarrier_spacing',15e3)
        self.subcarrier_num = kwargs.get('subcarrier_num',32)
        self.frequencies = subcarrier_frequencies(self.subcarrier_num, self.subcarrier_spacing)
        # 多普勒参数 ---------------------------------------------------------
        self.doppler_params = {
            "sampling_frequency": self.subcarrier_spacing,
            "num_time_steps": kwargs.get('num_time_steps',14),
            "target_velocities": kwargs.get('target_velocity',None)
        }
        # MUSIC估计参数 ---------------------------------------------------------
        self.music_params = {
            "start": kwargs.get('start',0),
            "end": kwargs.get('end',2000),
            "step": kwargs.get('step',0.5)
        } 
        # 初始化环境 ---------------------------------------------------------
        self.scene = self.mk_sionna_env()
        paths = self.scene.compute_paths(**self.ray_tracing_params)
        paths.normalize_delays = False
        paths.apply_doppler(**self.doppler_params)
        a,tau = paths.cir()
        self.h_env = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        # 获取环境图片
        self.scene_image = self._get_scene_image(filename = "scene.png")
        self.normalized_scene_image = self._normalize_image(self.scene_image)
        
        del paths,a,tau
        # 特征数量 ---------------------------------------------------------
        # self.feature_with_target = kwargs.get('feature_with_target',False)
        # (针对距离估计，特征为不同子载波上的信道信息)num_BS * num_BS * num_subcarrier * (real+img) + pos_now + velocity_now
        # if self.feature_with_target:
            # self.n_features = self.h_env.shape[1]**2 * self.h_env.shape[6] * 2 + 6
        # else:
            # self.n_features = self.h_env.shape[1]**2 * self.h_env.shape[6] * 2
        # (针对距离估计，特征为场景图像信息和目标位置坐标) image_size_x * image_size_y * 3 + 3
        self.n_features = 1
        
    def mk_sionna_env(self,tg=None,tgname=None,tgv=None,empty=False,test=False):
        if tg is None:
            scene = load_scene(self.env_path)
        else:
            scene = load_sensing_scene(self.env_path,tg,empty=empty)
        #配置天线阵列------------------------------------------------
        scene.tx_array = PlanarArray(**self.tx_params)
        scene.rx_array = PlanarArray(**self.rx_params)
        scene.frequency = self.frequency # in Hz; implicitly updates RadioMaterials
        scene.synthetic_array = self.synthetic_array # If set to False, ray tracing will be done per antenna element (slower for large arrays)
        # if self.BS_pos_trainable:
        #     self.BS_pos = [tf.Variable(pos) for pos in self.BS_pos]
        # 添加目标接收端用于辅助估计----------------------------------
        if test:
            rx = Receiver(name='rx-target',position = self.pos_now)
            scene.add(rx)
        for idx in range(self.BS_num):
            pos = self.BS_pos[idx]
            tx = Transmitter(name=f'tx{idx}',position=pos)
            rx = Receiver(name=f'rx{idx}',position=pos)
            scene.add(tx)
            scene.add(rx)
        
        #配置场景材质属性--------------------------------------------
        p1 = LambertianPattern()
        p2 = DirectivePattern(20)
        if scene.get("itu_plywood") is not None:
            scene.get("itu_plywood").scattering_coefficient = 0.3
            scene.get("itu_plywood").scattering_pattern = p1
        if scene.get("itu_concrete") is not None:
            scene.get("itu_concrete").scattering_coefficient = 0.5
            scene.get("itu_concrete").scattering_pattern = p1
        if scene.get("itu_glass") is not None:
            scene.get("itu_glass").scattering_coefficient = 0.25
            scene.get("itu_glass").scattering_pattern = p2
        if scene.get("itu_medium_dry_ground") is not None:
            scene.get("itu_medium_dry_ground").scattering_coefficient = 0.8
            scene.get("itu_medium_dry_ground").scattering_pattern = p1
        if scene.get("itu_metal") is not None:
            scene.get("itu_metal").scattering_coefficient = 0.1
            scene.get("itu_metal").scattering_pattern = p2
        #################配置感知目标#################
        if tgname is not None and tgv is not None:
            scene.target_names = tgname
            scene.target_velocities = tgv
        return scene
    
    def get_observation(self,observation_mode='pos'):
        '''obesrvation为场景图片+目标位置坐标
        '''
        # 判断基站和目标之间是否是视距
        self.scene = self.mk_sionna_env(test=True)
        self.paths = self.scene.compute_paths(**self.ray_tracing_params)
        self.los = self._is_los()
        
        # image = self._get_scene_image(filename = "observation.png") # 含目标的场景图片
        
        # 创建感知场景：只包含目标的场景
        target = Target(self.target_path, self.target_material, translate=self.pos_now)
        self.scene = self.mk_sionna_env(tg=target,tgname=[self.target_name],tgv=[self.velocity_now],empty=True)
        self.ray_tracing_params["scat_keep_prob"] = 1
        self.paths = self.scene.compute_paths(**self.ray_tracing_params)
        self.ray_tracing_params["scat_keep_prob"] = self.scat_keep_prob_fixed
        self.paths.normalize_delays = False
        self.doppler_params["target_velocities"] = self.scene.compute_target_velocities(self.paths)
        self.paths.apply_doppler(**self.doppler_params)
        a,tau = self.paths.cir()
        self.h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        
        # observation = self._normalize_image(image)
        # print(f"shape of scene_image: {self.scene_image.shape}")
        # print(f"shape of normalized_scene_image: {self.normalized_scene_image.shape}")
        
        # print(f"shape of observation: {observation.shape}")
        # observation = self.pos_now
        
        if observation_mode == 'pos':
            observation = tf.constant(self.pos_now,dtype=tf.float32)
        elif observation_mode == 'image':
            pass
        elif observation_mode == 'both':
            observation = tf.concat([self.normalized_scene_image,tf.constant(self.pos_now,dtype=tf.float32)],axis=0)
            pass
        
        return observation
    
    def reset(self):
        '''
            重置场景
        '''
        self.next_end = False # 用于标记一轮模拟结束
        self.pos_list = []    # 用于存储目标移动路径
        self.step_count = 0   # 用于记录当前步数
        # 生成目标移动路径, (目标移动策略: 'random','graph')
        if MOVE_STRATEGY == 'random':
            # 只在平面移动
            while np.random.rand() < 0.5 or len(self.pos_list) < 2: # 以0.5的概率继续移动,或者至少移动1次
                pos_now = np.zeros(3)
                pos_now[0] = (np.random.rand()-0.5)*DASX
                pos_now[1] = (np.random.rand()-0.5)*DASY
                self.pos_list.append(pos_now)
        elif MOVE_STRATEGY == 'graph':
            self.start = True
            # 当前位置
            # 随机选择一个起始位置
            pos_now_idx = np.random.randint(0,self.num_start_points)
            pos_now = self.start_points[pos_now_idx]
            # 偏移
            x_bias = (np.random.rand()*2-1)*self.point_bias
            pos_now[0] = pos_now[0] + x_bias
            y_bias = (np.random.rand()*2-1)*self.point_bias
            pos_now[1] = pos_now[1] + y_bias
            # 下一位置
            self.pos_list.append(pos_now)
            while True:
                has_path = np.where(self.point_path[pos_now_idx])[0]
                pos_next_idx = has_path[np.random.randint(0,len(has_path))]
                if self.start:
                    pos_next = self.points[pos_next_idx-self.num_start_points,:]
                    self.start = False
                else:
                    if pos_next_idx >= self.num_start_points+self.num_points:
                        # 终点
                        pos_next = self.end_points[pos_next_idx-self.num_start_points-self.num_points,:]
                        self.next_end = True
                    else:
                        # 移动点
                        pos_next = self.points[pos_next_idx-self.num_start_points,:]
                x_bias = (np.random.rand()*2-1)*self.point_bias
                pos_next[0] = pos_next[0] + x_bias
                y_bias = (np.random.rand()*2-1)*self.point_bias
                pos_next[1] = pos_next[1] + y_bias
                self.pos_list.append(pos_next)
                pos_now_idx = pos_next_idx
                if self.next_end:
                    break
        # 生成初始位置和速度、状态
        # print(f"Position list: {self.pos_list}")
        self.path_len = len(self.pos_list)
        self.next_pos_idx = 1
        self.pos_now = self.pos_list[self.next_pos_idx-1]
        pos_dis = self.pos_list[self.next_pos_idx]-self.pos_list[self.next_pos_idx-1]
        self.velocity_now = (pos_dis)/(np.linalg.norm((pos_dis))) * np.random.rand() * VCRG[1] # 单位向量*速度*随机范围
        if np.linalg.norm(self.velocity_now) < VCRG[0]:
            self.velocity_now = self.velocity_now / np.linalg.norm(self.velocity_now) * VCRG[0]
        # 设置场景，获取CSI
        target = Target(self.target_path, self.target_material, translate=self.pos_now)
        self.scene = self.mk_sionna_env(tg=target,tgname=[self.target_name],tgv=[self.velocity_now])
        observation = self.get_observation()
        self.reward = 0
        self.done = False
        return observation
    
    def step(self, action):
        #奖励 reward------------------------------------------------------------------------------------
        self.reward = self._get_reward(action,method='crb')[action]
        self.step_count = self.step_count + 1
        # 目标移动-----------------------------------------------------------------------------------
        move_length = np.linalg.norm(self.velocity_now * TIME_SLOT)
        rest_length = np.linalg.norm(self.pos_list[self.next_pos_idx]-self.pos_now)
        if move_length >= rest_length:
            self.pos_now = self.pos_list[self.next_pos_idx]
            self.next_pos_idx += 1
            if self.next_pos_idx == self.path_len: # 当前要到达的点是最后一个点
                self.done = True
            else:
                pos_dis = self.pos_list[self.next_pos_idx]-self.pos_list[self.next_pos_idx-1]
                self.velocity_now = pos_dis/(np.linalg.norm(pos_dis)) * np.linalg.norm(self.velocity_now)# 变更速度方向
        else:
            self.pos_now = self.pos_now + self.velocity_now * TIME_SLOT
        # 超出边界
        if self.pos_now[0]>=DASX or self.pos_now[0]<=-DASX or self.pos_now[1]>=DASY or self.pos_now[1]<=-DASY:
            self.done=True
        # 速度随机变化-----------------------------------------------------------------------------------
        if np.random.rand() < VCRT:
            self.velocity_now = self.velocity_now * (((np.random.rand()*2-1)*VCS + np.linalg.norm(self.velocity_now))/np.linalg.norm(self.velocity_now))
            if np.linalg.norm(self.velocity_now) < VCRG[0]:
                self.velocity_now = self.velocity_now / np.linalg.norm(self.velocity_now) * VCRG[0]
            elif np.linalg.norm(self.velocity_now) > VCRG[1]:
                self.velocity_now = self.velocity_now / np.linalg.norm(self.velocity_now) * VCRG[1]
        # 下一次state-----------------------------------------------------------------------------------
        tg = Target(self.target_path, self.target_material, translate=self.pos_now)
        self.scene = self.mk_sionna_env(tg=tg,tgname=[self.target_name],tgv=[self.velocity_now])
        self.next_observation = self.get_observation()            
        return self.next_observation, self.reward, self.done 
    
    def get_data_label(self,target_info=False):
        # 判断基站和目标之间是否是视距
        self.scene = self.mk_sionna_env(test=True)
        self.paths = self.scene.compute_paths(**self.ray_tracing_params)
        self.los = self._is_los()
        # image = self._get_scene_image(filename = "scene_with_target.png") # 含目标的场景图片
        # 创建感知场景：只包含目标的场景
        target = Target(self.target_path, self.target_material, translate=self.pos_now)
        self.scene = self.mk_sionna_env(tg=target,tgname=[self.target_name],tgv=[self.velocity_now],empty=True)
        self.ray_tracing_params["scat_keep_prob"] = 1
        self.paths = self.scene.compute_paths(**self.ray_tracing_params)
        self.ray_tracing_params["scat_keep_prob"] = self.scat_keep_prob_fixed
        self.paths.normalize_delays = False
        self.doppler_params["target_velocities"] = self.scene.compute_target_velocities(self.paths)
        self.paths.apply_doppler(**self.doppler_params)
        a,tau = self.paths.cir()
        self.h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        label = 0
        crbs = self._get_reward(action=0,method='crb')
        self.reward = self._get_reward(action=0,method='crb')
        mses = []
        for action in range(self.BS_num):
            mse = self._get_reward(action=action,method='mse')
            mses.append(mse)
        if not target_info:
            data = self.h_freq[0,:,0,:,0,:,:]
            data = tf.transpose(data,perm=[2,3,0,1])
            data = tf.linalg.diag_part(data)
        else:
            data = tf.concat([tf.constant(self.pos_now,dtype=tf.float32),tf.constant(self.velocity_now,dtype=tf.float32)],axis=0)
        # data = tf.transpose(data,perm=[2,0,1])
        # data = tf.reshape(data,[-1,self.doppler_params["num_time_steps"],self.subcarrier_num])
        # data = tf.transpose(data,perm=[1,2,0])
        # 目标移动-----------------------------------------------------------------------------------
        # 移动距离 = 目标当前速度 * 时间间隔
        move_length = np.linalg.norm(self.velocity_now * TIME_SLOT)
        # 剩余移动距离 = 目标当前位置到下一个点的距离
        rest_length = np.linalg.norm(self.pos_list[self.next_pos_idx]-self.pos_now)
        if move_length >= rest_length:
            self.pos_now = self.pos_list[self.next_pos_idx]
            self.next_pos_idx += 1
            if self.next_pos_idx == self.path_len: # 当前要到达的点是最后一个点
                self.done = True
            else:
                pos_dis = self.pos_list[self.next_pos_idx]-self.pos_list[self.next_pos_idx-1]
                self.velocity_now = pos_dis/(np.linalg.norm(pos_dis)) * np.linalg.norm(self.velocity_now)# 变更速度方向
        else:
            self.pos_now = self.pos_now + self.velocity_now * TIME_SLOT
        # 超出边界
        if self.pos_now[0]>=DASX or self.pos_now[0]<=-DASX or self.pos_now[1]>=DASY or self.pos_now[1]<=-DASY:
            self.done=True
        # 速度随机变化-----------------------------------------------------------------------------------
        if np.random.rand() < VCRT:
            self.velocity_now = self.velocity_now * (((np.random.rand()*2-1)*VCS + np.linalg.norm(self.velocity_now))/np.linalg.norm(self.velocity_now))
            if np.linalg.norm(self.velocity_now) < VCRG[0]:
                self.velocity_now = self.velocity_now / np.linalg.norm(self.velocity_now) * VCRG[0]
            elif np.linalg.norm(self.velocity_now) > VCRG[1]:
                self.velocity_now = self.velocity_now / np.linalg.norm(self.velocity_now) * VCRG[1]
        
        if target_info:
            crbs = tf.constant(crbs,dtype=tf.float32)
            mses = tf.constant(mses,dtype=tf.float32)
            return tf.concat([data,tf.constant(self.pos_now,dtype=tf.float32),tf.constant(self.velocity_now,dtype=tf.float32)],axis=0),\
                tf.concat([crbs,mses],axis=0),self.done
        return data,label,self.done
    
    def get_data(self):
        # 数据为[pos_now, crbs, pos_next]
        # 判断基站和目标之间是否是视距
        self.scene = self.mk_sionna_env(test=True)
        self.paths = self.scene.compute_paths(**self.ray_tracing_params)
        self.los = self._is_los()
        # 创建感知场景：只包含目标的场景
        target = Target(self.target_path, self.target_material, translate=self.pos_now)
        self.scene = self.mk_sionna_env(tg=target,tgname=[self.target_name],tgv=[self.velocity_now],empty=True)
        self.ray_tracing_params["scat_keep_prob"] = 1
        self.paths = self.scene.compute_paths(**self.ray_tracing_params)
        self.ray_tracing_params["scat_keep_prob"] = self.scat_keep_prob_fixed
        self.paths.normalize_delays = False
        self.doppler_params["target_velocities"] = self.scene.compute_target_velocities(self.paths)
        self.paths.apply_doppler(**self.doppler_params)
        a,tau = self.paths.cir()
        self.h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True) # 计算频域响应
        crbs = self._get_crbs()
        
        pos_now = tf.constant(self.pos_now,dtype=tf.float32)
        # 目标移动-----------------------------------------------------------------------------------
        # 移动距离 = 目标当前速度 * 时间间隔
        move_length = np.linalg.norm(self.velocity_now * TIME_SLOT)
        # 剩余移动距离 = 目标当前位置到下一个点的距离
        rest_length = np.linalg.norm(self.pos_list[self.next_pos_idx]-self.pos_now)
        if move_length >= rest_length:
            self.pos_now = self.pos_list[self.next_pos_idx]
            self.next_pos_idx += 1
            if self.next_pos_idx == self.path_len: # 当前要到达的点是最后一个点
                self.done = True
            else:
                pos_dis = self.pos_list[self.next_pos_idx]-self.pos_list[self.next_pos_idx-1]
                self.velocity_now = pos_dis/(np.linalg.norm(pos_dis)) * np.linalg.norm(self.velocity_now)# 变更速度方向
        else:
            self.pos_now = self.pos_now + self.velocity_now * TIME_SLOT
        # 超出边界
        if self.pos_now[0]>=DASX or self.pos_now[0]<=-DASX or self.pos_now[1]>=DASY or self.pos_now[1]<=-DASY:
            self.done=True
        # 速度随机变化-----------------------------------------------------------------------------------
        if np.random.rand() < VCRT:
            self.velocity_now = self.velocity_now * (((np.random.rand()*2-1)*VCS + np.linalg.norm(self.velocity_now))/np.linalg.norm(self.velocity_now))
            if np.linalg.norm(self.velocity_now) < VCRG[0]:
                self.velocity_now = self.velocity_now / np.linalg.norm(self.velocity_now) * VCRG[0]
            elif np.linalg.norm(self.velocity_now) > VCRG[1]:
                self.velocity_now = self.velocity_now / np.linalg.norm(self.velocity_now) * VCRG[1]
        pos_next = tf.constant(self.pos_now,dtype=tf.float32)
        
        crbs = tf.constant(crbs,dtype=tf.float32)
        data = tf.concat([pos_now,crbs,pos_next],axis=0)
        return data, self.done

    def _get_scene_image(self,image_save_path=image_save_path,filename=None,
                         camera_position=[0,0,1000],camera_look_at=[0,0,0]):
        '''获取场景图片, 返回场景RGB图片数组
        '''
        if self.scene.get("render_view") is not None:
            self.scene.remove('render_view')
        render_camera = Camera("render_view",position=camera_position,look_at=camera_look_at)
        self.scene.add(render_camera)
        if os.path.exists(image_save_path) == False:
            os.makedirs(image_save_path)
        if filename is None:
            raise ValueError('filename is None')
        filename = image_save_path+filename
        self.scene.render_to_file(camera="render_view",
                                    filename=filename,
                                    resolution=IMAGE_RESOLUTION)
        scene_image = Image.open(filename)
        scene_image = np.asarray(scene_image)
        scene_image = scene_image[:,:,:3]
        scene_image = np.transpose(scene_image, [1, 0, 2])
        return scene_image
    
    def _normalize_image(self,image):
        '''return static environment image
        '''
        image = tf.convert_to_tensor(image) # 将numpy数组转换为张量
        image = tf.cast(image,tf.float32)
        image = image/255.0
        # image = tf.transpose(image, perm=[2,0,1]) # 转置
        # # image_flatten = tf.reshape(image,[-1,image.shape[1]*image.shape[2]])
        # image_flatten = tf.reshape(image,[-1])
        return image
    
    def _normalize_h(self,h):
        # h:[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, subcarrier_num]
        # h:[num_rx,num_tx,num_time_steps,subcarrier_num]
        h = h[0,:,0,:,0,:,:]
        # h:[feature_layers_num,num_time_steps,subcarrier_num]
        h_flatten = tf.reshape(h,[-1,self.doppler_params["num_time_steps"],self.subcarrier_num])
        h_real = tf.math.real(h_flatten)
        h_img = tf.math.imag(h_flatten)
        h_flatten = tf.concat([h_real,h_img],axis=0)
        h_flatten = tf.transpose(h_flatten,perm=[1,2,0])
        h_flatten = tf.reshape(h_flatten,[-1])
        return h_flatten

    def _music_range(self,h_freq,BS_id,frequencies,start = 0,end = 2000,step = 0.2):
        try:
            y_i = h_freq[0,BS_id,0,BS_id,0,0,:]
            y_i = tf.squeeze(y_i)
            y_i = tf.expand_dims(y_i, axis=0)
            y_i_H = tf.transpose(tf.math.conj(y_i))
            y_conv = tf.matmul(y_i_H, y_i)
            _, eig_vecs = tf.linalg.eigh(y_conv)
            tau_range = np.arange(start,end, step)
            G_n = tf.cast(eig_vecs[:,:-1], dtype=tf.complex64)
            G_n_H = tf.math.conj(tf.transpose(G_n))
            frequencies_c = tf.expand_dims(frequencies, axis=0)
            frequencies_c = tf.repeat(frequencies_c, len(tau_range), axis=0)
            frequencies_c = tf.cast(frequencies_c, dtype=tf.complex64)
            tau_range = tf.expand_dims(tau_range, axis=-1)
            tau_range = tf.repeat(tau_range, self.subcarrier_num, axis=-1)
            tau_range = tf.cast(tau_range, dtype=tf.complex64)
            a_m = tf.math.exp(-1j * 2 * np.pi * frequencies_c * (tau_range/1e9))
            a_m_H = tf.math.conj(tf.transpose(a_m))
            a_m_H = tf.expand_dims(a_m_H, axis=1)
            a_m_H = tf.transpose(a_m_H, perm=[2,0,1])
            a_m = tf.expand_dims(a_m, axis=1)
            G_n = tf.expand_dims(G_n, axis=0)
            G_n_H = tf.expand_dims(G_n_H, axis=0)
            P = 1 / (a_m @ G_n @ G_n_H @ a_m_H)
            P = tf.squeeze(P)
            # 计算谱函数
            P_tau_real = tf.math.real(P)
            P_tau_imag = tf.math.imag(P)
            P_abs = tf.math.sqrt(P_tau_real**2 + P_tau_imag**2)
            # P_norm = 10 * tf.math.log(P_abs / tf.reduce_max(P_abs), 10)
            # P_norm = tf.squeeze(P_norm)
            max_idx = tf.argmax(P_abs)
            range_est = (start + int(max_idx) * step)*0.15
            return range_est
        except:
            print("can't estimate!")
            return 0
          
    def _get_reward(self,action,method='mse',crb_target=None):
        '''奖励函数
        '''
        # 如果估计值和真实值的相差在真实值的5%以内，那么依据误差大小奖励在0~1之间
        # 否则，惩罚值在-1~0之间
        if method == 'mse':
            self.range_true = np.linalg.norm(self.BS_pos[action,:] - self.pos_now)
            if self.los[action]:
                self.range_est = self._music_range(self.h_freq,action,self.frequencies,**self.music_params) 
                diff = np.abs(self.range_true-self.range_est)
                diff = diff - self.target_size
                if diff < 0 :
                    diff = 0
                return diff
            else:
                self.range_est = 0
                return -1
        elif method == 'crb':
            if crb_target is None:
                raise ValueError('crb_target is None')
            # print(crb_target)
            c = -np.log10(crb_target)
            c = linear_normalize(c,1)
            # c_sorted = np.sort(c)
            c_sorted = np.flip(np.sort(c),axis=0)
            # print(c_sorted) 
            # c_max = np.max(c)
            # rewards = np.where(c == c_max,c,-c)
            
            rewards = c
            return rewards
    
    def _get_crbs(self):
        mask = self.scene.get_obj_mask(self.paths,singleBS=True)[0]
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps]
        crb = self.paths.crb_delay(diag=True,mask = mask)
        crb_target = tf.where(mask, crb, 1)
        a = tf.where(mask,self.paths.a,0)
        # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx,num_tx]
        a = tf.transpose(a,perm=[0,2,4,5,6,1,3])
        # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx]
        a = tf.linalg.diag_part(a)
        # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx,1]
        a = tf.expand_dims(a, axis=-1)
        a = tf.transpose(a,perm=[0,5,1,6,2,3,4])
        a = tf.abs(a)
        crb_target = tf.reduce_min(crb_target, axis=6)
        crb_target = tf.reduce_min(crb_target, axis=4)
        crb_target = tf.reduce_min(crb_target, axis=2)
        a = tf.reduce_max(a, axis=6)
        a = tf.reduce_max(a, axis=4)
        a = tf.reduce_max(a, axis=2)
        a_sortidx = tf.argsort(a, axis=-1, direction='DESCENDING')
        a_max_idx = tf.gather(a_sortidx, 0, axis=-1)
        a_max_idx = tf.reshape(a_max_idx, [-1])
        crb_target = tf.gather(crb_target, a_max_idx, axis=-1)
        crb_target = tf.reshape(crb_target, [-1,a_max_idx.shape[0]])
        crb_target = tf.linalg.diag_part(crb_target)
        crb_target = tf.reshape(crb_target, [a.shape[0], a.shape[1], a.shape[2]])
        crb_target = tf.squeeze(crb_target)
        return crb_target
    
    def _is_los(self):
        # 范围LoS的mask
        # [batch_size,max_num_paths]
        types = self.paths.types
        types = types[0,:]
        types = tf.squeeze(types)
        # [max_num_paths]
        los = tf.where(types == 0, True, False)
        los = tf.expand_dims(los, axis=-1)
        # los = tf.repeat(los, self.BS_num, axis=-1)
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] 
        masks = self.paths.mask
        if self.synthetic_array:
            masks = tf.transpose(masks, perm=[0,3,1,2])
            masks = masks[0,:,0,:]
        else:
            masks = tf.transpose(masks, perm=[0,5,2,4,1,3])
            masks = masks[0,:,:,:,0,:]
            masks = tf.reduce_any(masks, axis=2)
            masks = tf.reduce_any(masks, axis=3)
        masks = tf.squeeze(masks)
        # masks: [max_num_paths, num_tx]
        los = tf.logical_and(los, masks)
        los = tf.reduce_any(los, axis=0)
        return los.numpy()

def run(fixed_route=False):
    step = 0
    for episode in range(3000):
        datas = np.load(data_path)
        np.random.shuffle(datas)          # 以第一维度随机打乱训练数据
        datas = datas[:RL.batch_size,:]   # 从打乱后的训练数据中取出 batch_size 条 
        correct_action = 0
        RL.reward_sum = 0
        print(f"====={episode}th episode start=====")
        for inner_step in range(RL.batch_size):
            data = datas[inner_step]
            pos_now = data[:3]
            pos_next = data[-3:]
            crbs = data[3:-3]
            # 只传入目标位置y坐标进行训练，并进行归一化
            action, action_type = RL.choose_action(np.array(pos_now[1]/400.), crbs) # 选择动作
            # action, action_type = RL.choose_action(pos_now, crbs) # 选择动作
            reward = env._get_reward(action,method='crb',crb_target=crbs)[action]
            RL.reward_sum += reward
            if reward == 1.:
                correct_action += 1
            print(f"\r【{step}-{inner_step}th step】pos:[{float(pos_now[0]):.1f},{float(pos_now[1]):.1f},{float(pos_now[2]):.2f}]\tBS:{action}({action_type})\treward:{reward:.4f}")
            RL.store_transition(np.array(pos_now[1]/400.), action, reward, np.array(pos_next[1]/400.))
            # RL.store_transition(pos_now, action, reward, pos_next)
            step += 1
        RL.reward_sum_his = np.append(RL.reward_sum_his, RL.reward_sum)
        correct_action_rate = correct_action/RL.batch_size
        RL.correct_action_rate_his = np.append(RL.correct_action_rate_his, correct_action_rate)
        print(f"correct action rate:{correct_action_rate*100:.2f}%")
        RL.learn()

def store_data(fixed_route=False):
    '''存储数据，数据保存格式 pos_now,crb(7),pow_next，3+7+3
        fixed_route: 是否固定轨迹
        sample_num: 采样次数
    '''
    datas = None
    if os.path.exists(data_path): # 如果文件存在，则加载文件
        datas = np.load(data_path) 
    step = 0
    # 采样次数
    if fixed_route:
        sample_num = 1
    else:
        sample_num = 3000
    for episode in range(sample_num): # 目标沿直线以恒定速率移动，每次移动的轨迹相同，只采集一次数据
        print(f"====={episode}th episode start=====")
        env.reset() # 重置环境
        inner_step = 0
        while True:
            data, done = env.get_data()
            print(f"\r【{step}-{inner_step}th step】pos:[{float(env.pos_now[0]):.1f},{float(env.pos_now[1]):.1f},{float(env.pos_now[2]):.2f}]\tv:{np.linalg.norm(env.velocity_now):.2f}\t{env.los}")
            data = np.expand_dims(data,axis=0)
            if datas is None:
                datas = data
            else:
                datas = np.concatenate((datas,data),axis=0)
            if done:
                step += 1
                break
            inner_step += 1
        np.save(data_path,datas)
    print(f"=====data collection finished=====")
   
def test():
    data = np.load(data_path)
    np.random.shuffle(data)
    print(data[:,1])    # pos_now  
    print(data[:,-2])   # pos_next
    print(data.shape)
    
def linear_normalize(data, mode=0):
    '''线性归一化
        mode = 0: 归一化到[0,1]
        mode = 1: 归一化到[-1,1]
    '''
    min = np.min(data)
    max = np.max(data)
    if mode == 0:
        data = (data-min)/(max-min) # 归一化到[0,1]
    else:
        data = ((data-min)/(max-min)-0.5)*2 # 归一化到[-1,1]
    return data
           
if __name__ == "__main__":
    # 数据保存路径
    data_path = './datas/datas-0530.npy'
    # label_path = './datas/labels-0513.npy'
    # image_path = './datas/images-0513.npy'
    np.set_printoptions(precision=1) # 设置打印精度为一位小数
    # end_points = np.array([[0,-170,0.05],[0,170,0.05]])
    # points = np.array([])
    # point_path = np.array([[0,1],[1,0]])
    env = Environment() # 初始化环境
    print(env.x_range,env.y_range)
    model_save_path = f'./models/one_way_street/dqn-pos-0530/' # 模型保存路径
    # 初始化 DQN 网络
    RL = DQN(env.n_features,env.action_space,memory_size=1000,best_action=True,best_prob=0.9,reward_decay=0.9,batch_size=256,replace_target_iter=5)
    if os.path.exists(data_path) == False:
        store_data(fixed_route=True) # 采集数据
    # test()
    # run(fixed_route=True)        # 训练     