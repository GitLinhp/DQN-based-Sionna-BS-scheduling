import json

def config_load(filepath):
    '''加载配置文件
        filepath: 配置文件路径
    '''
    with open(filepath, 'r', encoding='utf-8') as file:  
        configs = json.load(file)
    env_configs = configs['sensing_env']
    target_configs = configs['sensing_target']
    return env_configs,target_configs