#%%
# import
import numpy as np

#%%
# hyper parameters pools
# dataset_name_list = ['2018-12-31_18', '2019-01-31_18', '2019-02-28_18', '2019-03-31_18', '2019-04-30_18', '2019-05-31_18', '2019-06-30_18']
dataset_name_list = ['2018-12-31_6', '2019-01-31_6', '2019-02-28_6', '2019-03-31_6', '2019-04-30_6', '2019-05-31_6', '2019-06-30_6']
num_factors_list = [400] #[100, 200, 400, 800]
num_epochs_list = [1000] # [200, 500, 1000]
learning_rate_list = [0.04] # [0.02, 0.04, 0.08]
lambda_reg_list = [0.0005] #[0.0005, 0.0002, 0.0001]

search_method = 'grid'
n_trial = 1000 # number of trials for random search
script = ''

hypp_list = []
for dataset_name in dataset_name_list:
    for num_factors in num_factors_list:
        for num_epochs in num_epochs_list:
            for learning_rate in learning_rate_list:
                for lambda_reg in lambda_reg_list:
                    tmp_dict = {'dataset_name': dataset_name, 'num_factors': num_factors, 'num_epochs': num_epochs, 'learning_rate': learning_rate, 'lambda_reg': lambda_reg}
                    hypp_list.append(tmp_dict)

#%%
# random search
if search_method == 'random':
    np.random.shuffle(hypp_list)
    hypp_list = hypp_list[:n_trial]
    total_epochs = 0
    for i in range(len(hypp_list)):
        script += 'echo {}/{}\n'.format(i+1, len(hypp_list))
        script += 'python main.py {} --num_factors {} --num_epochs {} --learning_rate {} --lambda_reg {}\n'.format(hypp_list[i]['dataset_name'], hypp_list[i]['num_factors'], hypp_list[i]['num_epochs'], hypp_list[i]['learning_rate'], hypp_list[i]['lambda_reg'])
        total_epochs += hypp_list[i]['num_epochs']

#%%
# grid search
if search_method == 'grid':
    total_epochs = 0
    for i in range(len(hypp_list)):
        script += 'echo {}/{}\n'.format(i+1, len(hypp_list))
        script += 'python main.py {} --num_factors {} --num_epochs {} --learning_rate {} --lambda_reg {}\n'.format(hypp_list[i]['dataset_name'], hypp_list[i]['num_factors'], hypp_list[i]['num_epochs'], hypp_list[i]['learning_rate'], hypp_list[i]['lambda_reg'])
        total_epochs += hypp_list[i]['num_epochs']
        
#%%
print('Total trials: {}'.format(len(hypp_list)))
print('Total epochs: {}'.format(total_epochs))
print('Total time: {}h'.format(round((len(hypp_list) * 2 + total_epochs*(26/1000)/60)/60, 2)))

#%%
# print and save script
# print(script)

fp = open("run.bat", "w")
fp.write(script)
fp.close()
# %%
