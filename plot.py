#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# find best score of each models
model_list = ['LightGCN', 'BiVAE', 'BPR']
dataset_name_18_list = ['2018-12-31_18', '2019-01-31_18', '2019-02-28_18', '2019-03-31_18', '2019-04-30_18', '2019-05-31_18', '2019-06-30_18']
dataset_name_6_list = ['2018-12-31_6', '2019-01-31_6', '2019-02-28_6', '2019-03-31_6', '2019-04-30_6', '2019-05-31_6', '2019-06-30_6']
scores_18 = {}
scores_6 = {}
scores_18['SMORe-BPR'] = [0.0906, 0.0654, 0.0535, 0.0554, 0.0764, 0.0660, 0.0301]
scores_6['SMORe-BPR'] = [0.0834, 0.0671, 0.0558, 0.0533, 0.0825, 0.0702, 0.0304]

for model in model_list:
    key = '{}_{}'.format(model, '18')
    train_log = pd.read_csv('./{}/train_log.csv'.format(model))
    scores_18[key] = []
    scores_6[key] = []
    for dataset_name_18 in dataset_name_18_list:
        scores_18[key].append(train_log[(train_log['dataset_name'] == dataset_name_18) & (train_log['result'] == 'success')]['score'].max())
    for dataset_name_6 in dataset_name_6_list:
        scores_6[key].append(train_log[(train_log['dataset_name'] == dataset_name_6) & (train_log['result'] == 'success')]['score'].max())

x = np.arange(len(dataset_name_18_list))
width = 0.15
keys = list(scores_18.keys())
for i in range(len(keys)):
    plt.bar(x + i * width, scores_18[keys[i]], width, label=keys[i].split('_')[0])
plt.xticks(x + width / 2, dataset_name_18_list, rotation=30)
plt.ylim(0, 0.11)
plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1,1), loc='upper right')
plt.show()

x = np.arange(len(dataset_name_6_list))
width = 0.15
keys = list(scores_6.keys())
for i in range(len(keys)):
    plt.bar(x + i * width, scores_6[keys[i]], width, label=keys[i].split('_')[0])
plt.xticks(x + width / 2, dataset_name_6_list, rotation=30)
plt.ylim(0, 0.11)
plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1,1), loc='upper right')
plt.show()



#%%
students = ['Jack', 'Mary', 'Mike', 'David']
math_scores = [78, 67, 90, 81]
history_scores = [94, 71, 65, 88]
x = np.arange(len(students))
width = 0.3
plt.bar(x, math_scores, width, color='green', label='Math')
plt.bar(x + width, history_scores, width, color='blue', label='History')
plt.xticks(x + width / 2, students)
plt.ylabel('Math')
plt.title('Final Term')
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.show()

#%%