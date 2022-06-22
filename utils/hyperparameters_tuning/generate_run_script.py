#%%
# import
import numpy as np
from recommenders.utils.timer import Timer

SEARCH_METHOD = 'random' # 'grid'
N_TRIAL = 100 # number of trials in random search


#%%
# LightGCN
script_name = 'main.py'
hyperparameters_pool = {
    'dataset_name': ['2018-12-31_18'], #, '2019-01-31_18', '2019-02-28_18', '2019-03-31_18', '2019-04-30_18', '2019-05-31_18', '2019-06-30_18', '2018-12-31_6', '2019-01-31_6', '2019-02-28_6', '2019-03-31_6', '2019-04-30_6', '2019-05-31_6', '2019-06-30_6'], 
    'embed_size': [1024],
    'n_layers': [1],
    'batch_size': [5000],
    'epochs': [10],
    'learning_rate': [0.00002],
    'decay': [0],
}

#%%
# generate all configs pool
def build_config_and_append(hyperparameter_pools: dict, key_index: int, configs_list: list, config: dict):
    if key_index < len(hyperparameter_pools.keys()):
        key = list(hyperparameter_pools.keys())[key_index]
        for value in list(hyperparameter_pools[key]):
            config[key] = value
            configs_list = build_config_and_append(hyperparameter_pools, key_index + 1, configs_list, config)
        return configs_list
    else:
        configs_list.append(config.copy())
        return configs_list

with Timer() as t:
    configs_pool = build_config_and_append(hyperparameters_pool, 0, [], {})
print("Took {} seconds for generating configs pool.".format(t))

#%%
script = ''

def generate_cmd(config: dict):
    cmd = ''
    cmd += 'python {} '
    for key in config.keys():
        cmd += '--{} {} '.format(key, config[key])
    return cmd

# random search
if SEARCH_METHOD == 'random':
    np.random.shuffle(configs_pool)
    configs_pool = configs_pool[:N_TRIAL]
    total_epochs = 0
    for i in range(len(configs_pool)):
        script += 'echo {}/{}\n'.format(i+1, len(configs_pool))
        script += generate_cmd(configs_pool[i]) + '\n'
        # script += 'python main.py {} --embed_size {} --n_layers {} --batch_size {} --epochs {} --learning_rate {} --decay {}\n'.format(configs_pool[i]['dataset_name'], configs_pool[i]['embed_size'], configs_pool[i]['n_layers'], configs_pool[i]['batch_size'], configs_pool[i]['epochs'], configs_pool[i]['learning_rate'], configs_pool[i]['decay'])
        total_epochs += configs_pool[i]['epochs']

# grid search
if SEARCH_METHOD == 'grid':
    total_epochs = 0
    for i in range(len(configs_pool)):
        script += 'echo {}/{}\n'.format(i+1, len(configs_pool))
        script += generate_cmd(configs_pool[i]) + '\n'
        # script += 'python main.py {} --embed_size {} --n_layers {} --batch_size {} --epochs {} --learning_rate {} --decay {}\n'.format(configs_pool[i]['dataset_name'], configs_pool[i]['embed_size'], configs_pool[i]['n_layers'], configs_pool[i]['batch_size'], configs_pool[i]['epochs'], configs_pool[i]['learning_rate'], configs_pool[i]['decay'])
        total_epochs += configs_pool[i]['epochs']
        
#%%
print('Total trials: {}'.format(len(configs_pool)))
print('Total epochs: {}'.format(total_epochs))
# print('Total time: {}h'.format(round(total_epochs*0.2/60, 2)))

#%%
# print and save script
# print(script)

fp = open("./run.bat", "w")
fp.write(script)
fp.close()
# %%
