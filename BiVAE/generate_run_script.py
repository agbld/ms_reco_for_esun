#%%
dataset_name_list = ['2018-12-31_18'] #, '2019-01-31_18', '2019-02-28_18', '2019-03-31_18', '2019-04-30_18', '2019-05-31_18', '2019-06-30_18']
latent_dim_list = [64, 128, 256]
encoder_dims_list = [64, 128, 256]
act_func_list = ['sigmoid'] #, 'tanh'] #, 'elu', 'relu', 'relu6']
likelihood_list = ['pois'] #, 'bern'] #, 'gaus']
num_epochs_list = [20, 40, 80, 120]
batch_size_list = [1024]
learning_rate_list = [0.001, 0.002, 0.004]

script = ''

count_now = 0
count_all = 0

for i in range(2):
    for dataset_name in dataset_name_list:
        for latent_dim in latent_dim_list:
            for encoder_dims in encoder_dims_list:
                for act_func in act_func_list:
                    for likelihood in likelihood_list:
                        for num_epochs in num_epochs_list:
                            for batch_size in batch_size_list:
                                for learning_rate in learning_rate_list:
                                    if i == 0:
                                        count_all += 1
                                    else:
                                        count_now += 1
                                        script += 'echo {}/{}\n'.format(count_now, count_all)
                                        script += 'python main.py {dataset_name} --latent_dim {latent_dim} --encoder_dims {encoder_dims} --act_func {act_func} --likelihood {likelihood} --num_epochs {num_epochs} --batch_size {batch_size} --learning_rate {learning_rate}\n'.format(dataset_name=dataset_name, latent_dim=latent_dim, encoder_dims=encoder_dims, act_func=act_func, likelihood=likelihood, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)

print(script)

fp = open("run.bat", "w")
fp.write(script)
fp.close()
# %%
