r'''
通过路径计算CRB
'''

import tensorflow as tf
import numpy as np
from sionna.utils.tensors import insert_dims
from sionna.rt.paths import Paths

CHECK = False
# 检查 open3d 是否安装
if CHECK:
    try:
        import open3d as o3d
        open3d_installed = True
    except:
        open3d_installed = False


def crb_delay(paths:Paths,snr=10,diag=False,masks=None):
    r"""
    计算时延估计crb
    
    Input
    -----
    paths: :class:`~sionna.rt.paths.Paths`
        路径
    snr: (int, optional)
        信噪比. 默认为10dB.
    diag: (bool, optional)
        如果设置为`True`，则返回crb矩阵的对角线. 默认为`False`,
        建议仅在单个基站感知的情况下将diag设置为`True`
        
    Output
    -----
    crb: float32 
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps]
        时延估计CRB，若路径不存在则设置为1
    """
    # [batch_size,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths,num_time_steps]
    a = paths._a
    # [batch_size,num_rx,num_tx,max_num_paths]
    tau = paths._tau 
    
    if masks is not None:
        if tf.reduce_sum(tf.cast(masks,tf.int32)) == 0:
            # 掩码全零
            return tf.repeat(tf.zeros_like(a,dtype=tf.int8), repeats=len(masks), axis=0)
        else:
            # 相关参数
            num_rx = a.shape[1]         # 接收机数
            num_rx_ant = a.shape[2]     # 接收天线数
            if diag:
                num_tx = 1
            else:
                num_tx = a.shape[3]     # 发射机数
            num_tx_ant = a.shape[4]     # 发射天线数
            max_num_paths = a.shape[5]  # 最大路径数
            num_time_steps = a.shape[6] # 时间步数
            frequency = paths._scene.frequency  # 载波频率
            
            if paths._scene.synthetic_array:
                tau = tf.expand_dims(tau, axis=3)
                tau = tf.expand_dims(tau, axis=2)  
            
            crbs = []   # 用于保存crb
                        
            for idx in range(len(masks)):
                a_masked = tf.where(masks[idx], a, tf.zeros_like(a))         
                
                # 返回对角线元素
                if diag: 
                    # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,,num_rx,num_tx]
                    tau = tf.transpose(tau,perm=[0,2,4,5,1,3])
                    # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_rx]
                    tau = tf.linalg.diag_part(tau)
                    # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_rx,1]
                    tau = tf.expand_dims(tau, axis=-1)
                    # [batch_size,num_rx,num_rx_ant,1,num_tx_ant,max_num_paths]
                    tau = tf.transpose(tau,perm=[0,4,1,5,2,3])
                    # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx,num_tx]
                    a_masked = tf.transpose(a_masked,perm=[0,2,4,5,6,1,3])
                    # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx]
                    a_masked = tf.linalg.diag_part(a_masked)
                    # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx,1]
                    a_masked = tf.expand_dims(a_masked, axis=-1)
                    # [batch_size,num_rx,num_rx_ant,1,num_tx_ant,max_num_paths,num_time_steps]
                    a_masked = tf.transpose(a_masked,perm=[0,5,1,6,2,3,4])
                
                # [batch_size, num_rx, num_rx_ant/1, num_tx, num_tx_ant/1, max_num_paths*max_num_paths]
                tau_i = tf.repeat(tau,max_num_paths,axis=-1)
                # [batch_size, num_rx, num_rx_ant/1, num_tx, num_tx_ant/1, max_num_paths, max_num_paths]
                tau_i = tf.reshape(tau_i, [tau.shape[0],tau.shape[1],tau.shape[2],tau.shape[3],tau.shape[4],max_num_paths,max_num_paths])
                # [batch_size, num_rx, num_rx_ant/1, num_tx, num_tx_ant/1, max_num_paths, max_num_paths]
                tau_j = tf.transpose(tau_i,perm=[0,1,2,3,4,6,5])
                # [batch_size, num_rx, num_rx_ant/1, num_tx, num_tx_ant/1, max_num_paths, max_num_paths]
                tau_i_mine_j = tau_i- tau_j
                # [batch_size, num_rx, num_rx_ant/1, num_tx, num_tx_ant/1, max_num_paths, max_num_paths]
                tau_i_mul_j = tau_i* tau_j
                # [batch_size, num_rx, num_rx_ant/1, num_tx, num_tx_ant/1, max_num_paths, max_num_paths, 1]
                tau_i_mine_j = tf.expand_dims(tau_i_mine_j, axis=-1)
                # [batch_size, num_rx, num_rx_ant/1, num_tx, num_tx_ant/1, max_num_paths, max_num_paths, 1]
                tau_i_mul_j = tf.expand_dims(tau_i_mul_j, axis=-1)
                # [batch_size, num_rx, num_rx_ant/1, num_tx, num_tx_ant/1, 1, max_num_paths, num_time_steps]
                alpha = tf.expand_dims(a_masked, axis=-2)
                # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, 1, max_num_paths]
                alpha_1 = tf.transpose(alpha,perm=[0,1,2,3,4,7,5,6])
                # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, max_num_paths, 1]
                alpha_2 = tf.transpose(alpha,perm=[0,1,2,3,4,7,6,5])
                # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, max_num_paths, max_num_paths]
                alpha_ij = tf.matmul(alpha_1,alpha_2)
                # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, max_num_paths, num_time_steps]
                alpha_ij = tf.transpose(alpha_ij,perm=[0,1,2,3,4,6,7,5])
                one = tf.ones((max_num_paths,max_num_paths))
                one = insert_dims(one, 5, 0)
                # [1,1,1,1,1, max_num_paths, max_num_paths,1]
                one = insert_dims(one, 1, -1)
                F_alpha= 2*snr*tf.math.divide_no_nan(tf.math.abs(alpha_ij),(tau_i_mul_j**2))
                F_cos = (one+4*(np.pi**2)*(frequency) * tau_i_mul_j)*tf.math.cos(2*np.pi*frequency*tau_i_mine_j)
                F_sin = 2*np.pi*frequency*tau_i_mine_j*tf.math.sin(2*np.pi*frequency*tau_i_mine_j)
                F = F_alpha*(F_cos+F_sin)
                del alpha,alpha_1,alpha_2,alpha_ij,tau_i_mine_j,tau_i_mul_j,tau_i,tau_j,F_alpha,F_cos,F_sin,one
                # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, max_num_paths, max_num_paths]
                F = tf.transpose(F,perm=[0,1,2,3,4,7,5,6])
                F = tf.reshape(F, [-1,max_num_paths,max_num_paths])
                crb = tf.linalg.diag_part(tf.linalg.pinv(F))
                crb = tf.abs(crb)
                # 对于无效路径，将crb设置为1
                crb = tf.where(crb==0.0,1.0,crb)
                # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, max_num_paths]
                crb = tf.reshape(crb, [-1,num_rx,num_rx_ant,num_tx,num_tx_ant,num_time_steps,max_num_paths])
                # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps]
                crb = tf.transpose(crb,perm=[0,1,2,3,4,6,5])
                
                crbs.append(crb)
    return crbs

# 输出 crb 3d 点云，尚未进行适配  
def export_crb(paths:Paths,crb,filename:str,
                BS_pos = None,
                color_start = np.array([[60/255, 5/255, 80/255]]),
                color_mid = np.array([[35/255, 138/255, 141/255]]),
                color_end = np.array([[1, 1, 35./255]])):
    """
    输出 crb 3d 点云
    
    Input
    -----
    crb (_type_): 
        get from the method Paths.crb_delay
    filename (str): 
        recommend to use .xyzrgb as the suffix
    BS_pos (_type_, optional): 
        the position of the BS. Defaults to None.
    color_start (_type_, optional): 
        colorbar. Defaults to np.array([[60/255, 5/255, 80/255]]).
    color_mid (_type_, optional): 
        colorbar. Defaults to np.array([[35/255, 138/255, 141/255]]).
    color_end (_type_, optional): 
        colorbar. Defaults to np.array([[1, 1, 35./255]]).
    """
    objects = paths._objects
    vertices = paths._vertices
    mask = paths._mask
    num_rx = paths._a.shape[1]
    num_rx_ant = paths._a.shape[2]
    num_tx = paths._a.shape[3]
    num_tx_ant = paths._a.shape[4]
    max_num_paths = paths._a.shape[5]
    max_depth = objects.shape[0]
    num_targets = objects.shape[1]
    num_sources = objects.shape[2]
    
    # consider VH/cross-polarization
    if objects.shape[1] != num_rx*num_rx_ant:
        objects = tf.repeat(objects,int(num_rx*num_rx_ant/num_targets),axis=1)
    if objects.shape[2] != num_tx*num_tx_ant:
        objects = tf.repeat(objects,int(num_tx*num_tx_ant/num_sources),axis=2)
    objects = tf.reshape(objects, [max_depth,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths])
    
    if vertices.shape[1] != num_rx*num_rx_ant:
        vertices = tf.repeat(vertices,int(num_rx*num_rx_ant/num_targets),axis=1)
    if vertices.shape[2] != num_tx*num_tx_ant:
        vertices = tf.repeat(vertices,int(num_tx*num_tx_ant/num_sources),axis=2)
    vertices = tf.reshape(vertices, [max_depth,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths,3])
    
    if mask.shape[1] != num_rx*num_rx_ant:
        mask = tf.repeat(mask,int(num_rx*num_rx_ant/num_targets),axis=1)
    if mask.shape[2] != num_tx*num_tx_ant:
        mask = tf.repeat(mask,int(num_tx*num_tx_ant/num_sources),axis=2)
    mask = tf.reshape(mask, [-1,num_rx,num_rx_ant,num_tx,num_tx_ant,max_num_paths])
    # reduce num_time_steps dimension
    # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
    crb_ = tf.reduce_min(crb,axis=-1)
    
    crb_ = tf.where(mask,crb_,1)
    crb_ = tf.repeat(crb_,max_depth,axis=0)
    # mask out the paths that are valid
    indices = tf.where(objects != -1)
    
    # [valid_paths, 3]
    v = tf.gather_nd(vertices, indices)
    # [valid_paths]
    c = tf.gather_nd(crb_, indices)
    
    c = tf.where(c==0,1,c)
    c = tf.where(c==1,0,c)
    indices = tf.where(c != 0)
    c = tf.gather_nd(c, indices)
    v = tf.gather_nd(v, indices)
    c = np.log10(c)
    # c = np.abs(c)
    c = (c - np.min(c)) / (np.max(c) - np.min(c))
    
    c_color = np.expand_dims(c, axis=-1)
    c_color = np.repeat(c_color,3,axis=-1)
    
    color_start = np.repeat(color_start,c.shape[0],axis=0)
    color_mid = np.repeat(color_mid,c.shape[0],axis=0)
    color_end = np.repeat(color_end,c.shape[0],axis=0)
    
    c_color = np.where(c_color<0.5,color_start+(color_mid-color_start)*c_color*2,color_mid+(color_end-color_mid)*(c_color-0.5)*2)    
        
    if BS_pos is not None:
        BS_pos = np.array(BS_pos)
        BS_pos = np.expand_dims(BS_pos, axis=0)
        v = np.concatenate((v,BS_pos),axis=0)
        c_color = np.concatenate((c_color,np.array([[1,0,0]])),axis=0)
    else:
        v = v.numpy()
    
    if open3d_installed:
        print("open3d is installed, save the file as .xyzrgb")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v)
        pcd.colors = o3d.utility.Vector3dVector(c_color)
        return o3d.io.write_point_cloud(filename, pcd)
    else:
        print("open3d is not installed, save the file as .npy")
        try:
            np.save(f"{filename}-positions.npy",v)
            np.save(f"{filename}-colors.npy",c_color)
            return True
        except:
            return False