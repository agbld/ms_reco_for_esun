#%%
import numpy as np

# dataset_name_list = ['2018-12-31_18', '2019-01-31_18', '2019-02-28_18', '2019-03-31_18', '2019-04-30_18', '2019-05-31_18', '2019-06-30_18']
dataset_name_list = ['2018-12-31_18'] #, '2019-01-31_6', '2019-02-28_6', '2019-03-31_6', '2019-04-30_6', '2019-05-31_6', '2019-06-30_6']
latent_dim_list = [1024] #, 2048]
encoder_layers_list = [0, 1, 2]
act_func_list = ['sigmoid'] #, 'tanh'] #, 'elu', 'relu', 'relu6']
likelihood_list = ['pois'] #, 'bern'] #, 'gaus']
num_epochs_list = [120, 160, 250, 400] # [10, 20, 40, 80, 120]
batch_size_list = [2000, 4000] #, 8000]
learning_rate_list = [0.0005, 0.001, 0.002]

search_method = 'random'
n_trial = 200  # number of trials for random search
script = ''

hypp_list = []
for dataset_name in dataset_name_list:
    for latent_dim in latent_dim_list:
        for encoder_dims in encoder_layers_list:
            for act_func in act_func_list:
                for likelihood in likelihood_list:
                    for num_epochs in num_epochs_list:
                        for batch_size in batch_size_list:
                            for learning_rate in learning_rate_list:
                                tmp_dict = {'dataset_name': dataset_name, 'latent_dim': latent_dim, 'encoder_layers': encoder_dims, 'act_func': act_func, 'likelihood': likelihood, 'num_epochs': num_epochs, 'batch_size': batch_size, 'learning_rate': learning_rate}
                                hypp_list.append(tmp_dict)
                                # if i == 0:
                                #     count_all += 1
                                # else:
                                #     count_now += 1
                                #     script += 'echo {}/{}\n'.format(count_now, count_all)
                                #     script += 'python main.py {dataset_name} --latent_dim {latent_dim} --encoder_dims {encoder_dims} --act_func {act_func} --likelihood {likelihood} --num_epochs {num_epochs} --batch_size {batch_size} --learning_rate {learning_rate}\n'.format(dataset_name=dataset_name, latent_dim=latent_dim, encoder_dims=encoder_dims, act_func=act_func, likelihood=likelihood, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)

# random search
if search_method == 'random':
    np.random.shuffle(hypp_list)
    hypp_list = hypp_list[:n_trial]
    total_epochs = 0
    for i in range(len(hypp_list)):
        script += 'echo {}/{}\n'.format(i+1, len(hypp_list))
        script += 'python main.py {} --latent_dim {} --encoder_layers {} --act_func {} --likelihood {} --num_epochs {} --batch_size {} --learning_rate {}\n'.format(hypp_list[i]['dataset_name'], hypp_list[i]['latent_dim'], hypp_list[i]['encoder_layers'], hypp_list[i]['act_func'], hypp_list[i]['likelihood'], hypp_list[i]['num_epochs'], hypp_list[i]['batch_size'], hypp_list[i]['learning_rate'])
        total_epochs += hypp_list[i]['num_epochs']

# grid search
if search_method == 'grid':
    total_epochs = 0
    for i in range(len(hypp_list)):
        script += 'echo {}/{}\n'.format(i+1, len(hypp_list))
        script += 'python main.py {} --latent_dim {} --encoder_layers {} --act_func {} --likelihood {} --num_epochs {} --batch_size {} --learning_rate {}\n'.format(hypp_list[i]['dataset_name'], hypp_list[i]['latent_dim'], hypp_list[i]['encoder_layers'], hypp_list[i]['act_func'], hypp_list[i]['likelihood'], hypp_list[i]['num_epochs'], hypp_list[i]['batch_size'], hypp_list[i]['learning_rate'])
        total_epochs += hypp_list[i]['num_epochs']

# print(script)
print('Total trials: {}'.format(len(hypp_list)))
print('Total epochs: {}'.format(total_epochs))
print('Total time: {}h'.format(round(len(hypp_list)* 3 / 60, 2)))

fp = open("run.bat", "w")
fp.write(script)
fp.close()
# %%
