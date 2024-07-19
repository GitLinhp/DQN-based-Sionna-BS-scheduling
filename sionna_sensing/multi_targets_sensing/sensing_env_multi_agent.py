r'''
Sionna 感知基站调度环境配置，兼容Sionna v0.18.0
'''

import os
import tensorflow as tf
import numpy as np
from sionna.channel import subcarrier_frequencies
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from sionna.rt.scattering_pattern import *
from sionna.utils.tensors import insert_dims
from sionna_sensing.sensing_target import Target
from sionna_sensing.sensing_paths import crb_delay, export_crb
from sionna_sensing.dqn_agent import Agent
import pandas as pd
import PIL.Image as Image
import xml.etree.ElementTree as ET

class Environment():
    def __init__(self,image_save_path,target_configs,args,**kwargs):
        self.scene = None
        # 环境路径
        self.scene_name = kwargs.get('scene_name')
        self.scene_path = f'./scenes/{self.scene_name.title()}/{self.scene_name}.xml'
        # 基站参数---------------------------------------------------------
        self.BS_pos = np.array(kwargs.get('BS_pos'))    # 基站位置，过高可能导致感知效果不佳
        self.BS_num = len(self.BS_pos)                  # 基站数量
        # 天线配置参数 ---------------------------------------------------------
        self.tx_params = kwargs.get('tx_params')                # 发射天线参数
        self.rx_params = kwargs.get('rx_params')                # 接收天线参数
        self.frequency = kwargs.get('frequency')                # 中心载波频率
        self.synthetic_array = kwargs.get('synthetic_array')    # 天线阵列同步状态
        self.BS_pos_trainable = kwargs.get('BS_pos_trainable')  # 基站位置是否可训练
        # 射线追踪参数 ---------------------------------------------------------
        self.ray_tracing_params = kwargs.get('ray_tracing_params')
        self.scat_keep_prob_fixed = self.ray_tracing_params["scat_keep_prob"]
        # 频域信道参数 ---------------------------------------------------------
        self.subcarrier_spacing = kwargs.get('subcarrier_spacing')
        self.subcarrier_num = kwargs.get('subcarrier_num')
        # self.frequencies = subcarrier_frequencies(self.subcarrier_num, self.subcarrier_spacing)
        # 多普勒参数 ---------------------------------------------------------
        self.doppler_params = {
            "sampling_frequency": self.subcarrier_spacing,
            "num_time_steps": kwargs.get('doppler_params').get('num_time_steps'),
        }
        # MUSIC估计参数 ---------------------------------------------------------
        self.music_params = kwargs.get('music_params')
        # 其他参数
        self.DASX = kwargs.get('DASX')                          # X 轴限制移动范围
        self.DASY = kwargs.get('DASY')                          # Y 轴限制移动范围
        self.TIME_SLOT = kwargs.get('TIME_SLOT')                # 采样时间间隔
        self.IMAGE_RESOLUTION = kwargs.get('IMAGE_RESOLUTION')  # 图像分辨率
        self.image_save_path = image_save_path                  # 图像保存路径
        # 目标参数 ---------------------------------------------------------
        self.target_num = len(target_configs) 
        self.target_config(target_configs)
        self.pos_now = np.zeros([self.target_num,3])
        # 初始化环境 ---------------------------------------------------------
        self.scene = self.mk_sionna_env()
        # 获取环境图片 ---------------------------------------------------------
        self.scene_image = self._get_scene_image(filename="scene.png")        # 场景RGB图片np数组
        self.normalized_scene_image = self._normalize_image(self.scene_image) # 归一化场景图片
        # DQN 参数 ---------------------------------------------------------
        self.episodes_number = args['episode_number']       # 模型训练的最大代数
        self.max_ts = args['max_timestep']                  # 
        self.test = args['test']
        self.filling_steps = args['first_step_memory']
        self.steps_b_updates = args['replay_steps']
        self.action_space = self.target_num+1               # 动作空间，选择感知目标序号（包含不选择目标）
        self.state_space = np.array([self.target_num, 2])   # 状态大小，各目标的x,y位置坐标
        
        self.num_agents = self.target_num

    def add_target(self,target:Target):
        r'''
        添加目标
        
        Input
        -----
        target: :class:`sionna_sensing.sensing_target.Target`
            感知目标
        '''
        if isinstance(target, Target) == False:
            raise ValueError('target must be a class Target')
        self.targets.append(target)
    
    def remove_target(self,target_name:str):
        r'''
        移除目标
        
        Input
        -----
        target: :class:`sionna_sensing.sensing_target.Target`
            感知目标
        '''
        for target in self.targets:
            if target.name == target_name:
                self.targets.remove(target)
                self.target_num -= 1
                break
        raise ValueError(f'target \'{target_name}\' not found')
    
    def get_target_list(self):
        r'''
        获取目标列表
        
        Output
        -----
        targets: list
            目标列表, :class:`sionna_sensing.sensing_target.Target`
        '''
        if self.targets is None:
            raise ValueError('no targets')
        return self.targets
    
    def target_config(self,target_configs):
        r'''
        目标配置
        
        Input
        -----
        target_configs: dict
            目标配置参数
        '''
        self.targets = []
        if self.target_num > 8:
            raise ValueError('The number of targets should be less than 8')
        for idx in range(self.target_num):
            # 创建目标
            target = Target(**target_configs.get(f'target_{idx+1}'))
            # 将目标添加到目标列表中
            self.add_target(target)
        
    def mk_sionna_env(self):
        r'''
        初始化Sionna环境
        
        Output
        -----
        scene: :class:`sionna.rt.Scene`
            Sionna场景
        '''
        scene = load_sensing_scene(self.scene_path,self.targets) # 载入含目标的场景
        # 配置目标
        for target in self.targets:
            target.SceneObject = scene.get(target.name)
            target.SceneObject.position = target.initial_position       # 初始位置
            target.SceneObject.orientation = target.initial_orientation # 初始朝向
        # 配置天线阵列------------------------------------------------
        scene.tx_array = PlanarArray(**self.tx_params)  # 配置发射天线阵列
        scene.rx_array = PlanarArray(**self.rx_params)  # 配置接收天线阵列
        scene.frequency = self.frequency                # 中心载波频率，以 Hz 为单位，隐含更新电磁材料特性
        scene.synthetic_array = self.synthetic_array    # 天线阵列同步状态，若设为真计算时间将大幅提高
        # 添加感知基站--------------------------------------------
        for idx in range(self.BS_num):
            pos = self.BS_pos[idx]
            tx = Transmitter(name=f'tx{idx}',position=pos)
            rx = Receiver(name=f'rx{idx}',position=pos)
            scene.add(tx)
            scene.add(rx)
        # 配置场景材质属性--------------------------------------------
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
        return scene
       
    def reset(self):
        r'''
        重置目标状态：生成目标移动路径,重置目标移动完成状态
        
        Output
        -----
        初始状态
        '''
        for idx, target in enumerate(self.targets):
            target.generate_motion_path(self.DASX,self.DASY) # 生成目标移动路径
            target.movement.done = False                     # 重置目标移动完成状态
            self.pos_now[idx] = target.pos_now
        self.done = [False]*self.target_num                  # 用于标记各个目标移动是否
        
        return self.get_state()
    
    def get_data(self,show_scene_image=False):
        '''
        获取训练数据，用于离线训练
        数据格式：self.target_num*[pos_now, crbs, pos_next]
            crbs: [self.target_num,self.BS_num]
        
        Input
        -----
        show_scene_image: bool, 默认为`False`
            是否保存场景图片
            
        Output
        -----
        data: np.array
            训练数据
        self.done: list
            目标移动完成状态
        '''
        self.paths = self.scene.compute_paths(**self.ray_tracing_params) # 计算路径
        crbs = self._get_crbs() # 获取 CRB
        # 获取场景图片
        if show_scene_image:
            self._get_scene_image(filename="scene_test.png") 
        # 目标移动-----------------------------------------------------------------------------------
        pos_now = np.zeros([self.target_num,3])
        pos_next = np.zeros([self.target_num,3])
        
        for idx,target in enumerate(self.targets):
            pos_now[idx],pos_next[idx],self.done[idx] = target.move(self.DASX,self.DASY,self.TIME_SLOT)

        data = tf.concat([pos_now,crbs,pos_next],axis=1)
        data = np.expand_dims(data, axis=0)
        
        return data, self.done

    def _is_los(self):
        r'''
        获取LoS径的掩码
            masks: [max_num_paths, num_rx, num_tx]
        
        Output
        -----
        los: np.array
            LoS径掩码
        '''
        # 添加辅助接收机用于判断是否基站与目标之间存在LoS径
        for idx,target in enumerate(self.targets):
            rx = Receiver(name=f'rx-target_{idx}',position=target.pos_now+[0,0,1.0])
            self.scene.add(rx)
        self.paths = self.scene.compute_paths(**self.ray_tracing_params)
        # types: [batch_size, max_num_paths]
        types = self.paths.types
        types = types[0,:]
        los = tf.where(types == 0, True, False)
        los = tf.expand_dims(los, axis=-1)
        # [max_num_paths, num_rx]
        los = tf.repeat(los, self.BS_num+self.target_num, axis=-1)  
        los = tf.expand_dims(los, axis=-1)  
        # [max_num_paths, num_rx, num_tx]
        los = tf.repeat(los, self.BS_num, axis=-1)  
        masks = self.paths.mask
        if self.synthetic_array:
            # 天线同步状态为 True
            # [batch_size, num_rx, num_tx, max_num_paths]
            masks = tf.transpose(masks, perm=[0,3,1,2])
            masks = masks[0,:,:,:]
        else:
            # 天线同步状态为 False
            # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] 
            masks = tf.transpose(masks, perm=[0,5,2,4,1,3])
            masks = masks[0,:,:,:,:,:]
            masks = tf.reduce_any(masks, axis=2)
            masks = tf.reduce_any(masks, axis=3)
        # [max_num_paths, num_rx, num_tx]
        masks = tf.squeeze(masks)   # 从张量的形状中移除大小为1的维度
        los = tf.logical_and(los, masks)    
        los = tf.reduce_any(los, axis=0)
        # 移除辅助估计接收端
        for idx in range(self.target_num):
            rx_name = f'rx-target_{idx}'
            self.scene.remove(rx_name)
        return los.numpy()
        
    def _get_scene_image(self,filename:str,camera_position=[0,0,1000],camera_look_at=[0,0,0]):
        r'''
        获取场景图片, 并返回场景RGB图片np数组
        
        Input
        -----
        filename: str
            场景图片保存路径
        camera_position: list, default=[0,0,1000]
            相机位置
        camera_look_at: list, default=[0,0,0]
            相机看向的位置
        
        Output
        -----
        scene_image: np.array
            场景图片RGB数组
        '''
        # 添加相机
        camera_name = "scene_image_cam"
        if self.scene.get(camera_name) is not None:
            self.scene.remove(camera_name)
        camera = Camera(camera_name,position=camera_position,look_at=camera_look_at)
        self.scene.add(camera)
        # 保存场景图片
        if os.path.exists(self.image_save_path) == False:
            os.makedirs(self.image_save_path)
        elif isinstance(filename,str) == False:
            raise ValueError('filename must be a string')
        filename = self.image_save_path+filename
        self.scene.render_to_file(camera=camera_name,
                                  filename=filename,
                                  resolution=self.IMAGE_RESOLUTION)
        # 读取场景图片，并转换为np数组
        scene_image = Image.open(filename)
        scene_image = np.asarray(scene_image)
        scene_image = scene_image[:,:,:3]
        scene_image = np.transpose(scene_image, [1, 0, 2])
        return scene_image
    
    def _normalize_image(self,image):
        r'''
        场景图片归一化
        
        Input
        -----
        image: np.array
            待归一化的场景图片np数组
        
        Output
        -----
        normalized_image: tf.Tensor
            归一化后的场景图片张量
        '''
        image = tf.convert_to_tensor(image) # 将numpy数组转换为张量
        image = tf.cast(image,tf.float32)
        normalized_image = image/255.0
        return normalized_image
    
    def _normalize_h(self,h):
        r'''
        信道归一化，尚未完成适配
        
        Input
        -----
        h: tf.Tensor
            信道脉冲响应张量
            
        Output
        -----
        h_flatten: tf.Tensor
            归一化后的信道脉冲响应张量
        '''
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
        r'''
        MUSIC距离估计，尚未完成适配
        
        Input
        -----
        h_freq: tf.Tensor
            信道频率响应张量
        BS_id: int
            基站ID
        frequencies: tf.Tensor
            频率张量
        start: int, default=0
            起始估计距禈
        end: int, default=2000
            终止估计距禈
        step: float, default=0.2
            步长
        '''
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
    
    def get_obj_mask(self,singleBS=True):
        r'''
        获取目标对象的路径掩码
        
        Input
        -----
            singleBS: bool, default=True 是否仅返回基站和目标间的路径掩码
            - 0 : LoS
            - 1 : Reflected
            - 2 : Diffracted
            - 3 : Scattered
        
        Output
        -----
        masks: list
            目标对象的路径掩码
        '''
        # 'objects' 名称和索引的字典
        obj_names = {}
        for i,s in enumerate(self.scene._scene.shapes()):
            # 名称格式: 'mesh-XX'
            name = s.id().split('-')
            if name[0] == 'mesh':
                name = name[1]
                obj_names[name] = i
        #------------------ 获取对象的掩码 -------------------
        # [max_depth,num_targets,num_sources,max_num_paths]
        objects = self.paths.objects
        # [max_num_wedges,2], 楔形
        wedges_2_objects = self.scene._solver_paths._wedges_objects
        # 如果目标和源之间的路径有效，则掩码为真
        # [1, num_targets, num_sources, max_num_paths]
        mask_tg_sr = self.paths.targets_sources_mask
        mask_tg_sr = tf.expand_dims(tf.expand_dims(mask_tg_sr, axis=-1), axis=0)
        # [max_num_paths]
        types = self.paths.types[0]
        # [1, 1, 1, max_num_paths]
        types = insert_dims(types, 3, 0)
        # 对象和楔形的掩码
        is_obj = tf.where(tf.logical_and(objects != -1,tf.logical_or(types == 1,types == 3)), True, False)
        is_wedge = tf.where(tf.logical_and(objects != -1,types == 2), True, False)
        is_obj_or_wedge = tf.logical_or(is_obj, is_wedge)
        
        num_rx = self.paths.a.shape[1]
        num_rx_ant = self.paths.a.shape[2]
        num_tx = self.paths.a.shape[3]
        num_tx_ant = self.paths.a.shape[4]
        max_num_paths = self.paths.a.shape[5]
        max_depth = objects.shape[0]
        
        # 将楔形转换为对象
        wedge1 = wedges_2_objects[:,0]
        wedge2 = wedges_2_objects[:,1]
        indices = tf.where(is_wedge)
        updates1 = tf.gather(wedge1, tf.reshape(objects[is_wedge], [-1]))
        updates2 = tf.gather(wedge2, tf.reshape(objects[is_wedge], [-1]))
        objects_wedge1 = tf.tensor_scatter_nd_update(objects, indices, updates1)
        objects_wedge2 = tf.tensor_scatter_nd_update(objects, indices, updates2)
        
        masks = []
        for target in self.targets:
            name = target.name
            idx = obj_names[name]
            # 与目标交互的路径掩码
            # [max_depth,num_targets,num_sources,max_num_paths]
            obj1_mask = tf.where(tf.logical_and(objects_wedge1==idx,is_obj_or_wedge), True, False)
            obj2_mask = tf.where(tf.logical_and(objects_wedge2==idx,is_obj_or_wedge), True, False)
            obj_mask = tf.logical_or(obj1_mask,obj2_mask)
            
            # [max_depth,num_targets,num_sources,max_num_paths,1]
            mask_paths = tf.expand_dims(obj_mask, axis=-1)
            # [max_depth,num_targets,num_sources,max_num_paths,1]
            mask = tf.logical_and(mask_tg_sr, mask_paths)
            if mask.shape[1] != num_rx*num_rx_ant:
                # 考虑交叉/水平垂直极化
                mask = tf.repeat(mask, repeats=int(num_rx*num_rx_ant/mask.shape[1]), axis=1)
            if mask.shape[2] != num_tx*num_tx_ant:
                # 考虑交叉/水平垂直极化
                mask = tf.repeat(mask, repeats=int(num_tx*num_tx_ant/mask.shape[2]), axis=2)
            # [max_depth,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths,1]
            mask = tf.reshape(mask, [max_depth,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths,1])
            # [1,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths,1]
            mask = tf.reduce_any(mask, axis=0, keepdims=True)
            if singleBS: # 只考虑本基站收发信机之间的路径
                # [1,num_rx_ant,num_tx_ant,max_num_paths,1,num_rx,num_tx]
                mask = tf.transpose(mask,perm=[0,2,4,5,6,1,3])
                # [1,num_rx_ant,num_tx_ant,max_num_paths,1,num_rx]
                mask = tf.linalg.diag_part(mask) # 获取对角阵元素
                # [1,num_rx_ant,num_tx_ant,max_num_paths,1,num_rx,1]
                mask = tf.expand_dims(mask, axis=-1)
                # [1,num_rx,num_rx_ant,1,num_tx_ant,max_num_paths,1]
                mask = tf.transpose(mask,perm=[0,5,1,6,2,3,4])
            masks.append(mask)
        return masks
    
    def step(self, action, crbs):
        r'''
        在线训练
        
        Input
        -----
        action: int
            动作
            
        Output
        -----
        self.next_observation: np.array
            下一步的观测值
        self.reward: float
            奖励值
        self.done: bool
            目标移动完成状态
        '''
        self.paths = self.scene.compute_paths(**self.ray_tracing_params)   # 计算路径
        crbs = self._get_crbs()
        #奖励 reward------------------------------------------------------------------------------------
        for idx in range(self.target_num):
            reward = self._get_reward(action,method='crb',crb_target=crbs)[idx,action] # 获取奖励
        # 目标移动-----------------------------------------------------------------------------------
        pos_now = np.zeros([self.target_num,3])
        pos_next = np.zeros([self.target_num,3])
        
        for idx,target in enumerate(self.targets):
            pos_now[idx],pos_next[idx],self.done[idx] = target.move(self.DASX,self.DASY,self.TIME_SLOT)
        self.pos_now = pos_now
        self.pos_next = pos_next
        self.state_ = self.get_state()            
        return self.state_, reward, self.done
    
    def _get_crbs(self):
        r'''
        返回每个基站对每个目标的时延估计CRB
        
        Output
        -----
        crb_target: np.array [num_BS,num_target]
            各基站对各目标的时延估计CRB
        '''
        # num_target*[1,num_rx,num_rx_ant,1,num_tx_ant,max_num_paths,1]
        masks = self.get_obj_mask(singleBS=True)
        # [batch_size,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths,num_time_steps]
        crb = crb_delay(self.paths,diag=True,masks=masks) # 获取时延估计的CRB  
        crb_target = tf.where(masks, crb, 1)              # 无效路径的CRB设置为1
        
        # [batch_size,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths,1]
        a = self.paths.a
        # [1,batch_size,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths,1]
        a = tf.expand_dims(a,axis=0)
        # [target_num,batch_size,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths,1]
        a = tf.repeat(a, repeats=self.target_num, axis=0)
        a = tf.where(masks,a,0)
        # [target_num,batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx,num_tx]
        a = tf.transpose(a,perm=[0,1,3,5,6,7,2,4])
        # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx]
        a = tf.linalg.diag_part(a)
        # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx,1]
        a = tf.expand_dims(a, axis=-1)
        a = tf.transpose(a,perm=[0,1,6,2,7,3,4,5])
        a = tf.abs(a)
        crb_target = tf.reduce_min(crb_target, axis=7)
        crb_target = tf.reduce_min(crb_target, axis=5)
        crb_target = tf.reduce_min(crb_target, axis=3)
        a = tf.reduce_max(a, axis=7)
        a = tf.reduce_max(a, axis=5)
        a = tf.reduce_max(a, axis=3)
        a_sortidx = tf.argsort(a, axis=-1, direction='DESCENDING')
        a_max_idx = tf.gather(a_sortidx, 0, axis=-1)
        a_max_idx = tf.reshape(a_max_idx, [-1])
        crb_target = tf.gather(crb_target, a_max_idx, axis=-1)
        crb_target = tf.reshape(crb_target, [-1,a_max_idx.shape[0]])
        crb_target = tf.linalg.diag_part(crb_target)
        crb_target = tf.reshape(crb_target, [a.shape[0], a.shape[1], a.shape[2], a.shape[3]])
        crb_target = tf.squeeze(crb_target)
        
        if self.target_num == 1:
            crb_target = tf.expand_dims(crb_target, axis=0)
        
        return crb_target
    
    def _get_reward(self,action,method='mse',crb_target:np.array=None):
        r'''
        获取奖励函数
        
        Input
        -----
        action: int
            动作
        method: str
            计算奖励的方法
                'mse': 均方误差
                'crb': CRB
        crb_target: np.array
            基站对目标感知的CRB
        
        Output
        -----
        rewards: float
            奖励
        '''
        if method == 'mse':
            # 需要修改
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
            
            c = -np.log10(crb_target)
            c = data_normalization(c,method='z_score')
            c = data_normalization(c,method='min_max')
            # [BS_num,target_num]
            rewards = c
            return rewards
    
    def get_state(self,state_mode='pos'):
        r'''
        获取观测值
        
        Input
        -----
        state_mode: str
            观测模式
                'pos': 位置坐标
                'image': 场景图片
                'both': 位置坐标+场景图片
        
        Output
        -----
        state : :class:`~tf.Tensor`
            观测值
        '''        
        if state_mode == 'pos':
            pos_now = self.pos_now
            pos_now = pos_now[:,:2]/[10.,400.]
            pos_now = tf.constant(pos_now,dtype=tf.float32)
            state = tf.reshape(pos_now,(-1))
        elif state_mode == 'image':
            pass
        elif state_mode == 'both':
            pass
        
        return state

    def run(self,agents:list):
        r'''
        环境运行
        
        Input
        -----
        agents : list of :class:`~sionna_sensing.dqn_agent.Agent` 
            智能体（基站）列表
        '''
        total_step = 0
        reward_sum_his = [] # 历史累积奖励
        max_score = -10000
        for episode in range(3000): 
            print(f"====={episode}th episode start=====")
            state = self.reset()    # 重置环境
            
            reward_sum = 0          # 记录每个episode的累计奖励
            time_step = 0
            while True:
                print(f"\r【{episode}-{time_step}th step】")
                # 显示目标信息：target_name[(pos_now),(velocity_now)]
                targets_info = "\t"
                for idx, target in enumerate(self.targets):
                    targets_info += f"target_{idx}[({float(target.pos_now[0]):.1f},{float(target.pos_now[1]):.1f},{float(target.pos_now[2]):.2f}),({target.velocity_now})] "
                print(targets_info)
                
                actions = []
                action_types = []
                actions_info = "\t"
                # 执行动作
                for idx, agent in enumerate(agents):
                    action, action_type = agent.greedy_actor(state)
                    actions.append(action)
                    action_types.append(action_type)
                    if idx != len(agents):
                        actions_info += f'{action}({action_type}),'
                    else:
                        actions_info += f'{action}({action_type})'
                state_, reward, done = self.step(actions)   # 更新环境，获取环境信息
                print(actions_info+f'\treward:{reward:.4f}')
                
                if not self.test:
                    for agent in agents:
                        # 获取全局观测
                        agent.observe((state, actions, reward, state_, done))
                        if total_step >= self.filling_steps:
                            agent.decay_epsilon()
                            if time_step % self.steps_b_updates == 0:
                                agent.replay()
                            agent.update_target_model()                   
                
                total_step += 1
                time_step += 1
                state = state_
                reward_sum += reward # 累计奖励
                
                if all(done): # 所有目标到达终点
                    break
                
            reward_sum_his.append(reward_sum)
            
            if not self.test:
                if episode % 5 == 0:
                    df = pd.DataFrame(reward_sum_his, columns=['score'])
                    df.to_csv(file1)

                    if total_step >= self.filling_steps:
                        if reward_sum > max_score:
                            for agent in agents:
                                agent.brain.save_model()
                            max_score = reward_sum
            

def data_normalization(data, method='min_max'):
    r'''
    数据归一化
    
    Input
    -----
    data: np.array
        待归一化的数据
    method: str
        归一化方法
            'min_max': 最小最大归一化
            'z_score': 零-均值归一化
            'log': 对数归一化
    
    Output
    -----
    normalized_data: np.array
        归一化后的数据
    '''
    if method == 'min_max':
        min = np.min(data)
        max = np.max(data)
        normalized_data = ((data-min)/(max-min)-0.5)*2
    elif method == 'z_score':
        mean = np.mean(data)
        std = np.std(data)
        normalized_data = (data-mean)/std
    elif method == 'log':
        normalized_data = np.log10(data)/np.log10(np.max(data))
    return normalized_data

def load_sensing_scene(filename,targets):
    r'''
    加载感知场景: 将目标添加到xml场景文件中，再加载Sionna场景
    
    Input
    -----
    filename: str
        场景文件路径
    targets: list
        目标列表
    dtype: tf.dtype, default=tf.complex64
        数据类型
        
    Output
    -----
    scene: :class:`~sionna.rt.Scene`
        Sionna场景
    '''
    root = ET.parse(filename).getroot() # 获取根目录
    if isinstance(targets, list):       # 传入的是目标列表
        if not all(isinstance(x, Target) for x in targets):
            raise ValueError('targets must be a list of class Target')
        for target in targets:
            xml = target_to_xml(target)
            root.append(xml)
    elif isinstance(targets, Target):   # 传入的是单个目标
        xml = target_to_xml(targets)
        root.append(xml)
    else:
        raise ValueError('targets must be a list of class Target or class Target')
    new_filename = filename.replace('.xml','_tmp.xml')
    with open(new_filename, 'wb') as f:
        f.write(ET.tostring(root))  
    return load_scene(new_filename)

def target_to_xml(target:Target):
    r"""
    将目标转换为Mitsuba使用的XML格式的字符串

    Input
    -----
    target : :class:`~sionna.rt.Target` 
        带转换的目标
    
    Output
    ------
    xml : str
        Mitsuba使用的XML格式的字符串
    """
    name = target.filename.split('.')[0]
    name = name.split('/')[-1]
    xml = f"""<shape type="ply" id="mesh-{name}" name="mesh-{name}">
    <string name="filename" value="{target.filename}"/>
    <boolean name="face_normals" value="true"/>
	<ref id="mat-{target.material}" name="bsdf"/>
    <transform name="to_world">
        <rotate x="{target.rotate[0]}" y="{target.rotate[1]}" z="{target.rotate[2]}" angle="{target.rotate[3]}"/>
        <scale x="{target.scale[0]}" y="{target.scale[1]}" z="{target.scale[2]}"/>
        <translate x="{target.translate[0]}" y="{target.translate[1]}" z="{target.translate[2]}"/>
    </transform>
    </shape>"""
    return ET.fromstring(xml)
