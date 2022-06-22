#%%
import sys
sys.path.append('../')

import os
import cornac
import papermill as pm
import scrapbook as sb
import pandas as pd
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer

import argparse
from datetime import datetime
from esun.esun_utils import Evaluation, get_esun_train_interactions, to_consecutive_id


print("System version: {}".format(sys.version))
print("Cornac version: {}".format(cornac.__version__))

#%%
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--num_factors', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lambda_reg', type=float, default=0.001)
    args = parser.parse_args()

    config = {'dataset_name': args.dataset_name, 
                'num_factors': args.num_factors,
                'num_epochs': args.num_epochs,
                'learning_rate': args.learning_rate,
                'lambda_reg': args.lambda_reg,
                'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    print('use cli')
except:
    config = {'dataset_name': '2018-12-31_18',
                'num_factors': 400,
                'num_epochs': 1000,
                'learning_rate': 0.04,
                'lambda_reg': 0.0005,
                'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    print('use jupyter')

print(config)

config['result'] = 'failed'
config['score'] = 0

if os.path.exists('./train_log.csv'):
    train_log = pd.read_csv('./train_log.csv')
    train_log = train_log.append(config, ignore_index=True)
else:
    train_log = pd.DataFrame([config])
train_log.to_csv('./train_log.csv', index=False)

#%%
train = get_esun_train_interactions(dataset_name=config['dataset_name'])
train['userID'], cust_no_2_userID = to_consecutive_id(train['cust_no'])
train['itemID'], wm_prod_code_2_itemID = to_consecutive_id(train['wm_prod_code'])
train = train[['userID', 'itemID', 'rating']]

#%%
# train, test = python_random_split(data, 0.75)

#%%
train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False))

print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))

#%%
bpr = cornac.models.BPR(
    k=config['num_factors'],
    max_iter=config['num_epochs'],
    learning_rate=config['learning_rate'],
    lambda_reg=config['lambda_reg'],
    verbose=True,
)

#%%
with Timer() as t:
    bpr.fit(train_set)
print("Took {} seconds for training.".format(t))

#%%
with Timer() as t:
    all_predictions = predict_ranking(bpr, train, usercol='userID', itemcol='itemID', remove_seen=False)
print("Took {} seconds for prediction.".format(t))

#%%
with Timer() as t:
    topk_scores = all_predictions.groupby('userID').apply(lambda x: x.nlargest(5, 'prediction')).droplevel(0)
print("Took {} seconds for picking top-k.".format(t))

userID_2_cust_no = {v: k for k, v in cust_no_2_userID.items()}
itemID_2_wm_prod_code = {v: k for k, v in wm_prod_code_2_itemID.items()}
topk_scores['cust_no'] = topk_scores['userID'].map(userID_2_cust_no)
topk_scores['wm_prod_code'] = topk_scores['itemID'].map(itemID_2_wm_prod_code)

recommendation = topk_scores.groupby('cust_no')['wm_prod_code'].apply(list).to_dict()

#%%
evaluation = Evaluation('', '../esun/dataset/{}/interaction_eval.csv'.format(config['dataset_name']), recommendation)
config['score'] = evaluation.results()
print('score: ', config['score'])

config['result'] = 'success'
config['end_time'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#%%
# score, results_per_user = evaluation.results_mod()
# log_name = 'BPR__{dataset_name}__{num_factors}__{num_epochs}__{learning_rate}__{lambda_reg}__{create_time}'.format(**config)
# results_per_user.to_csv('../results_per_user/{}.csv'.format(log_name), index=False)

#%%
train_log = pd.read_csv('./train_log.csv')
train_log.drop(train_log.tail(1).index,inplace=True)
train_log = train_log.append(config, ignore_index=True)
train_log.to_csv('./train_log.csv', index=False)

#%%