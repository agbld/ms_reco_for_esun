#%%
import sys
sys.path.append('../')

import os
from lightgbm import train
# import papermill as pm
# import scrapbook as sb
import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

import argparse

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams

from esun.esun_utils import Evaluation, get_esun_train_interactions, to_consecutive_id
from datetime import datetime

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))

#%%
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--embed_size', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_epoch', type=int, default=-1)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    config = {'dataset_name': args.dataset_name, 
            'embed_size': args.embed_size, 
            'n_layers': args.n_layers, 
            'batch_size': args.batch_size, 
            'decay': args.decay, 
            'epochs': args.epochs, 
            'eval_epoch': args.eval_epoch, 
            'learning_rate': args.learning_rate, 
            'top_k': args.top_k,
            'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    print('use cli')
except:
    config = {'dataset_name': '2018-12-31_18', 
            'embed_size': 1024, 
            'n_layers': 1, 
            'batch_size': 20000, 
            'decay': 0, #0.0001, 
            'epochs': 5, 
            'eval_epoch': -1, 
            'learning_rate': 0.00002, 
            'top_k': 5,
            'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    print('use jupyter')

print(config)
yaml_file = "./lightgcn.yaml"

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
train = train[['userID', 'itemID', 'rating', 'timestamp']]

#%%
# train, test = python_stratified_split(df, ratio=0.75)

#%%
data = ImplicitCF(train=train, test=None)

#%%

hparams = prepare_hparams(yaml_file,
                          embed_size=config['embed_size'],
                          n_layers=config['n_layers'],
                          batch_size=config['batch_size'],
                          decay=config['decay'],
                          epochs=config['epochs'],
                          eval_epoch=config['eval_epoch'],
                          learning_rate=config['learning_rate'],
                          top_k=config['top_k'],
                         )
# hparams = prepare_hparams(yaml_file,
#                           n_layers=config['n_layers'],
#                           batch_size=config['batch_size'],
#                           epochs=config['epochs'],
#                           learning_rate=config['learning_rate'],
#                           eval_epoch=config['eval_epoch'],
#                           top_k=config['top_k'],
#                          )

#%%
model = LightGCN(hparams, data)

#%%
with Timer() as train_time:
    model.fit()

print("Took {} seconds for training.".format(train_time.interval))

#%%
topk_scores = model.recommend_k_items(train, top_k=config['top_k'], remove_seen=False, use_id=True)

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
train_log = pd.read_csv('./train_log.csv')
train_log.drop(train_log.tail(1).index,inplace=True)
train_log = train_log.append(config, ignore_index=True)
train_log.to_csv('./train_log.csv', index=False)

#%%