import argparse
import math
 
# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
# DQN 参数
parser.add_argument('-e', '--episode-number', default=1000000, type=int, help='训练代数，Number of episodes')
parser.add_argument('-l', '--learning-rate', default=0.00005, type=float, help='学习率，Learning rate')
parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='RMSProp',
                        help='优化方法，Optimization method')
parser.add_argument('-m', '--memory-capacity', default=1000000, type=int, help='内存容量，Memory capacity')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='批次大小，Batch size')
parser.add_argument('-t', '--target-frequency', default=10000, type=int,
                    help='Number of steps between the updates of target network')
parser.add_argument('-x', '--maximum-exploration', default=100000, type=int, help='Maximum exploration step')
parser.add_argument('-fsm', '--first-step-memory', default=0, type=float,
                    help='Number of initial steps for just filling the memory')
parser.add_argument('-rs', '--replay-steps', default=4, type=float, help='更新网络之间的步数，steps between updating the network')
parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
parser.add_argument('-tt', '--target-type', choices=['DQN', 'DDQN'], default='DDQN', help='目标网络类型')
parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='PER',help='经验回放库类型')
parser.add_argument('-pl', '--prioritization-scale', default=0.5, type=float, help='Scale for prioritization')
parser.add_argument('-du', '--dueling', action='store_true', help='Enable Dueling architecture if "store_false" ')

parser.add_argument('-gn', '--gpu-num', default='3', type=str, help='训练使用的 GPU 编号')
parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase if "store_false"')
# 解析参数
args = parser.parse_args()
print(args)
args = vars(parser.parse_args())
print(args)
