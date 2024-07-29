import json

def config_load(filepath):
    r'''
    加载配置文件，为 json 格式
        
    Input
    -----
    filepath: str
        配置文件路径
        
    Output
    -----
    env_configs: dict
        环境配置
    target_configs: dict
        目标配置
    '''
    
    filetype = filepath.split('.')[-1]
    if filetype != 'json':
        raise("config file must be a 'json' file")
    with open(filepath, 'r', encoding='utf-8') as file:  
        configs = json.load(file)
    env_configs = configs['sensing_env']
    target_configs = configs['sensing_target']
    return env_configs,target_configs