import pandas as pd
from tqdm import tqdm
import os
import numpy as np
        
class Evaluation:
    
    def __init__(self, date, path, pred):
        self.today = date
        self.path = path
        self.pred = pred
        self.ans = self.answer(self.path)
    
    def show(self):
        print(f"Date: {self.today}\n") 
        coverage = len(set(self.pred.keys()) & set(self.ans.keys()))       
        print(f"Uppper-Bound: {coverage}\n")
 
    def answer(self, path):
        df = self.read(path)
        return df.groupby('cust_no')['wm_prod_code'].apply(list).to_dict()

    def read(self, path):
        return pd.read_csv(path, usecols=['cust_no', 'wm_prod_code'])  

    def results(self):
        p = 0
        count = len(self.ans)
        for u, pred in tqdm(self.pred.items(), total=count):
            p += self.precision_at_5(u, pred)
        return p/count
    
    def precision_at_5(self, user, pred):
        try:
            y_true = self.ans[user]
            tp = len(set(y_true) & set(pred))
            return tp/5
        except:
            return 0

def get_esun_train_interactions(dataset_name:str = '2018-12-31_6', overwrite=False):
    folder_path = '../esun/dataset/' + dataset_name + '/'
    if os.path.exists('../esun/dataset/' + dataset_name + '/interaction_train.csv') and not overwrite:
        inter_df = pd.read_csv('../esun/dataset/' + dataset_name + '/interaction_train.csv')
    else:
        files_list = []
        for f in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, f)) and (f.find('train_') >= 0): 
                files_list.append(os.path.join(folder_path, f))
        inter_df = []
        for file in files_list:
            print(file)
            df = pd.read_csv(file, usecols=['cust_no', 'wm_prod_code', 'txn_dt', 'deduct_cnt'])
            df['timestamp'] = pd.to_datetime(df['txn_dt'])
            df['timestamp'] = df.timestamp.values.astype(np.int64) // 10 ** 9
            inter_df.append(df)
        inter_df = pd.concat(inter_df)
        inter_df['rating'] = 1
        inter_df = inter_df[['cust_no', 'wm_prod_code', 'rating', 'timestamp']]
        inter_df.drop_duplicates(['cust_no', 'wm_prod_code'], inplace=True)
        inter_df.to_csv('../esun/dataset/' + dataset_name + '/interaction_train.csv', index=False)
    
    return inter_df

def to_consecutive_id(series: pd.Series, start_from = 0):
    id_list = list(series.unique())
    map_dict = {}
    id_tmp = start_from
    for id in id_list:
        map_dict[id] = id_tmp
        id_tmp += 1
    return series.map(map_dict), map_dict