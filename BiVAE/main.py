#%%
import sys
sys.path.append('../')

import os
import torch
import cornac
import papermill as pm
import scrapbook as sb
import pandas as pd
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

import argparse
from datetime import datetime
from esun.esun_utils import Evaluation, get_esun_train_interactions, to_consecutive_id

print("System version: {}".format(sys.version))
print("PyTorch version: {}".format(torch.__version__))
print("Cornac version: {}".format(cornac.__version__))
print('Torch CUDA available: {}'.format(torch.cuda.is_available()))

# print time
print('{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

#%%
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--encoder_layers', type=int, default=16)
    parser.add_argument('--act_func', type=str, default='tanh')
    parser.add_argument('--likelihood', type=str, default='pois')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    config = {'dataset_name': args.dataset_name, 
              'latent_dim': args.latent_dim,
              'encoder_layers': args.encoder_layers,
              'act_func': args.act_func,
              'likelihood': args.likelihood,
              'num_epochs': args.num_epochs,
              'batch_size': args.batch_size,
              'learning_rate': args.learning_rate,
              'top_k': args.top_k,
              'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    print('use cli')
except:
    # config = {'dataset_name': '2018-12-31_18',
    #           'latent_dim': 1024,
    #           'encoder_layers': 3,
    #           'act_func': 'sigmoid',
    #           'likelihood': 'pois',
    #           'num_epochs': 1,
    #           'batch_size': 8000,
    #           'learning_rate': 0.002,
    #           'top_k': 5,
    #           'create_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    config = {'dataset_name': '2018-12-31_18',
              'latent_dim': 1024,
              'encoder_layers': 0,
              'act_func': 'sigmoid',
              'likelihood': 'pois',
              'num_epochs': 5,
              'batch_size': 4000,
              'learning_rate': 0.002,
              'top_k': 5,
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

# Select MovieLens data size: 100k, 1m, 10m, or 20m
# MOVIELENS_DATA_SIZE = '100k'

# top k items to recommend
TOP_K = config['top_k']

# Model parameters
LATENT_DIM = config['latent_dim']
ENCODER_LAYERS =  config['encoder_layers']
ACT_FUNC = config['act_func']
LIKELIHOOD = config['likelihood']
NUM_EPOCHS = config['num_epochs']
BATCH_SIZE = config['batch_size']
LEARNING_RATE = config['learning_rate']

#%%
# data = movielens.load_pandas_df(
#     size=MOVIELENS_DATA_SIZE,
#     header=["userID", "itemID", "rating"]
# )
# data.head()

train = get_esun_train_interactions(dataset_name=config['dataset_name'])
train['userID'], cust_no_2_userID = to_consecutive_id(train['cust_no'])
train['itemID'], wm_prod_code_2_itemID = to_consecutive_id(train['wm_prod_code'])
train = train[['userID', 'itemID', 'rating', 'timestamp']]

#%%
# train, test = python_random_split(data, 0.75)

#%%
train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False))

print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))

#%%
if train_set.num_users < train_set.num_items:
    first_layer_dim = train_set.num_users
else:
    first_layer_dim = train_set.num_items
encoder_structure = []
for i in range(ENCODER_LAYERS):
    dim = (first_layer_dim - LATENT_DIM) / (ENCODER_LAYERS + 1) * (ENCODER_LAYERS - i) + LATENT_DIM
    encoder_structure.append(int(dim))
bivae = cornac.models.BiVAECF(
    k=LATENT_DIM,
    encoder_structure=encoder_structure,
    act_fn=ACT_FUNC,
    likelihood=LIKELIHOOD,
    n_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    use_gpu=torch.cuda.is_available(),
    verbose=True
)

with Timer() as t:
    bivae.fit(train_set)
print("Took {} seconds for training.".format(t))

#%%
with Timer() as t:
    all_predictions = predict_ranking(bivae, train, usercol='userID', itemcol='itemID', remove_seen=False)
print("Took {} seconds for prediction.".format(t))

#%%
with Timer() as t:
    topk_scores = all_predictions.groupby('userID').apply(lambda x: x.nlargest(TOP_K, 'prediction')).droplevel(0)
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
# log_name = 'BiVAE__{dataset_name}__{latent_dim}__{encoder_dims}__{act_func}__{likelihood}__{num_epochs}__{batch_size}__{learning_rate}__{top_k}__{create_time}'.format(**config)
# results_per_user.to_csv('../results_per_user/{}.csv'.format(log_name), index=False)

#%%
train_log = pd.read_csv('./train_log.csv')
train_log.drop(train_log.tail(1).index,inplace=True)
train_log = train_log.append(config, ignore_index=True)
train_log.to_csv('./train_log.csv', index=False)

#%%
# print time
print('{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))