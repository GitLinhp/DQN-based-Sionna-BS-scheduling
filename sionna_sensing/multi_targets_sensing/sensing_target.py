r'''
兼容Sionna v0.18.0
'''

import tensorflow as tf
import numpy as np
from sionna.rt.scene_object import SceneObject

class Movement():
    r'''
    目标移动参数
    '''
    def __init__(self,**kwargs):
        self.move_strategy = kwargs.get('move_strategy')    # 目标移动策略
        self.vcrt = kwargs.get('vcrt')                      # 速度变化概率
        self.vcs = kwargs.get('vcs')                        # 速度变化范围
        self.vcrg = kwargs.get('vcrg')                      # 速度范围
        if self.move_strategy == 'graph':
            self.start = True
            # start_points: [num1,3] float,指定的起点
            self.start_points = np.array(kwargs.get('start_points'))
            self.num_start_points = len(self.start_points)
            # points: [num2,3] float,指定的移动点
            self.points = np.array(kwargs.get('points'))
            self.num_points = len(self.points)
            # end_points: [num3,3] float,指定的终点
            self.end_points = np.array(kwargs.get('end_points'))
            self.num_end_points = len(self.end_points)
            # point_bias: float,移动点偏移范围
            # 目标会把目的点设置为以 point 为中心，point_bias为半径的圆内的随机点
            self.point_bias = kwargs.get('point_bias')
            # point_path:[num1+num2+num3,num1+num2+num3] int 邻接矩阵, 表示点之间的路径关系（有向图表示）
            # 前num1个为start_points,中间num2个为points,最后num3个为end_points
            self.point_path = np.array(kwargs.get('point_path'))
            num_path = len(self.point_path)
            # 检查路径是否与点对应
            if num_path != self.num_start_points + self.num_points + self.num_end_points:
                raise ValueError('point_path must be a (num_start_points+num_points+num_end_points) x (num_start_points+num_points+num_end_points) matrix')
            self.done = False

class Target:
    r'''
    感知目标
    '''
    def __init__(self,**kwargs):
        self.name = kwargs.get('name')                          # 名称
        self.filename = 'meshes/'+self.name+'.ply'              # 文件名
        self.material = kwargs.get('material')                  # 材质
        self.translate = kwargs.get('translate',(0.,0.,0.))     # 平移
        self.scale = kwargs.get('scale',(1.,1.,1.))             # 缩放
        self.rotate = kwargs.get('rotate',(0.,0.,0.,0.))        # 旋转
        self.size = kwargs.get('size')                          # 尺寸
        self.initial_position = np.array(kwargs.get('initial_position'))          # 初始位置
        self.initial_orientation = np.deg2rad(kwargs.get('initial_orientation'))  # 初始朝向
        self.pos_now = self.initial_position                    # 当前位置
        self.orientation_now = self.initial_orientation         # 当前朝向
        self.velocity_now = None                                # 当前速度
        self.movement:Movement = Movement(**kwargs.get('movement')) # 运动参数
        self.SceneObject:SceneObject = None                         # 场景对象，类型为：'sionna.rt.scene_object.SceneObject'
    
    def generate_motion_path(self,dasx,dasy):
        r'''
        生成目标移动路径，为移动点列表
        
        目标移动策略 random,graph
        random:
            在区域内随机移动，需更改配置dasx,dasy，目标将在以原点为中心，dasx*dasy的矩形区域内随机移动
        graph: 
            按照指定路线移动,主要用于模拟车辆轨迹/路上行人轨迹。需构建环境时提供如下参数：
                start_points: [num1,3] float
                    指定的起点
                points: [num2,3] float
                    指定的移动点
                end_points: [num3,3] float
                    指定的终点
                point_bias: float
                    移动点偏移范围,目标会把目的点设置为以point为中心，point_bias为半径的圆内的随机点
                point_path:[num1+num2+num3,num1+num2+num3] int
                    邻接矩阵，表示点之间的路径关系（有向图表示）,前num1个为start_points,中间num2个为points,最后num3个为end_points
            
        Input
        -----
        dasx: float
            x方向的移动范围
        dasy: float
            y方向的移动范围
        
    '''     
        vcrg = self.movement.vcrg
        next_end = False            # 是否下一个点是终点
        self.pos_list = []          # 用于存储目标移动路径
        if self.movement.move_strategy == 'random':
            # 只在平面移动
            while np.random.rand() < 0.5 or len(self.pos_list) < 2: # 以0.5的概率继续移动,或者至少移动1次
                pos_now = np.zeros(3)
                pos_now[0] = (np.random.rand()-0.5)*dasx
                pos_now[1] = (np.random.rand()-0.5)*dasy
                pos_now[2] = 0.75
                self.pos_list.append(pos_now)
        elif self.movement.move_strategy == 'graph':
            # 随机选择一个起始位置
            pos_now_idx = np.random.randint(0,self.movement.num_start_points)
            pos_now = self.movement.start_points[pos_now_idx].copy()
            # 偏移
            x_bias = (np.random.rand()*2-1)*self.movement.point_bias
            pos_now[0] = pos_now[0] + x_bias
            y_bias = (np.random.rand()*2-1)*self.movement.point_bias
            pos_now[1] = pos_now[1] + y_bias
            # 将当前位置加入路径列表中
            self.pos_list.append(pos_now)
            # 下一位置
            while True:
                has_path = np.where(self.movement.point_path[pos_now_idx])[0]
                pos_next_idx = has_path[np.random.randint(0,len(has_path))]
                if pos_next_idx >= self.movement.num_start_points+self.movement.num_points:
                    # 终点
                    pos_next = self.movement.end_points[pos_next_idx-self.movement.num_start_points-self.movement.num_points,:]
                    next_end = True
                else:
                    # 移动点
                    pos_next = self.movement.points[pos_next_idx-self.movement.num_start_points,:]
                # 偏移
                x_bias = (np.random.rand()*2-1)*self.movement.point_bias
                pos_next[0] = pos_next[0] + x_bias
                y_bias = (np.random.rand()*2-1)*self.movement.point_bias
                pos_next[1] = pos_next[1] + y_bias
                self.pos_list.append(pos_next)
                pos_now_idx = pos_next_idx
                if next_end:
                    break
        # 生成初始位置和速度、状态
        self.path_len = len(self.pos_list)
        self.next_pos_idx = 1
        self.pos_now = self.pos_list[self.next_pos_idx-1]
        pos_dis = self.pos_list[self.next_pos_idx]-self.pos_list[self.next_pos_idx-1] # 移动距离
        self.velocity_now = (pos_dis)/(np.linalg.norm((pos_dis))) * (np.random.rand()*(vcrg[1]-vcrg[0])+vcrg[0])    # 单位向量*速度*随机范围
        
        self.SceneObject.position = self.pos_now        # 更新目标实例位置
        self.SceneObject.velocity = self.velocity_now   # 更新目标实例速度
        
    def move(self,dasx,dasy,time_slot):
        r'''
        目标移动
        
        Input
        -----
        dasx: float
            x方向的移动范围
        dasy: float
            y方向的移动范围
        time_slot: float
            采样时间间隔
        
        Output
        -----
        pos_now: [float,3]
            当前位置
        pos_next: [float,3]
            下一时刻位置
        done: bool
            是否完成移动
        '''
        if self.movement.done:
            pos_now = self.pos_now
            pos_next = pos_now
            return pos_now,pos_next,self.movement.done
        
        vcrt = self.movement.vcrt   # 速度变化概率
        vcrg = self.movement.vcrg   # 速度变化范围
        vcs = self.movement.vcs     # 速度变化大小
        
        pos_now = tf.constant(self.pos_now,dtype=tf.float32)
        # 目标移动-----------------------------------------------------------------------------------
        # 移动距离 = 目标当前速度 * 时间间隔
        move_length = np.linalg.norm(self.velocity_now * time_slot)
        # 剩余移动距离 = 目标当前位置到下一个点的距离
        rest_length = np.linalg.norm(self.pos_list[self.next_pos_idx]-self.pos_now)
        if move_length >= rest_length:
            self.pos_now = self.pos_list[self.next_pos_idx]
            self.next_pos_idx += 1
            if self.next_pos_idx == self.path_len: # 当前要到达的点是最后一个点
                self.movement.done = True
            else:
                pos_dis = self.pos_list[self.next_pos_idx]-self.pos_list[self.next_pos_idx-1]
                self.velocity_now = pos_dis/(np.linalg.norm(pos_dis)) * np.linalg.norm(self.velocity_now)# 变更速度方向
        else:
            self.pos_now = self.pos_now + self.velocity_now * time_slot
        # 超出边界
        if np.abs(self.pos_now[0])>=dasx or np.abs(self.pos_now[1])>=dasy:
            self.movement.done = True
        # 速度随机变化-----------------------------------------------------------------------------------
        if np.random.rand() < vcrt:
            self.velocity_now = self.velocity_now * (((np.random.rand()*2-1)*vcs + np.linalg.norm(self.velocity_now))/np.linalg.norm(self.velocity_now))
            if np.linalg.norm(self.velocity_now) < vcrg[0]:
                self.velocity_now = self.velocity_now / np.linalg.norm(self.velocity_now) * vcrg[0]
            elif np.linalg.norm(self.velocity_now) > vcrg[1]:
                self.velocity_now = self.velocity_now / np.linalg.norm(self.velocity_now) * vcrg[1]
        pos_next = tf.constant(self.pos_now,dtype=tf.float32)
        
        self.SceneObject.position = self.pos_now        # 更新目标实例位置
        self.SceneObject.velocity = self.velocity_now   # 更新目标实例速度
        
        return pos_now, pos_next, self.movement.done