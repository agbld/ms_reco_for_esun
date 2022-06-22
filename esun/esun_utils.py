#%%
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

def concat_datasets():
    full_dataset = None
    path = './dataset'
    folders = os.listdir(path)
    with tqdm(total=len(folders)) as pbar:
        for folder in folders:
            folder = os.path.join(path, folder)
            if not os.path.isdir(folder) or folder.find('_6') >= 0:
                # print(folder, 'is not included')
                pbar.update(1)
                continue
            # print('Processing folder: ' + folder)
            dataset = []
            for file in os.listdir(folder):
                file = os.path.join(folder, file)
                if os.path.isfile(file) and (file.find('train_') >= 0): 
                    # print('Loading file: ' + file)
                    tmp_df = pd.read_csv(file)
                    dataset.append(tmp_df)
            dataset = pd.concat(dataset)
            if full_dataset is None:
                full_dataset = dataset
            else:
                full_dataset = pd.concat([full_dataset, dataset], ignore_index=True)
                full_dataset.drop_duplicates(inplace=True)
            pbar.update(1)
    
    return full_dataset

#%%

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

    def results_mod(self):
        p = 0
        count = len(self.ans)
        results = []
        for u, pred in tqdm(self.pred.items(), total=count):
            p_at_5 = self.precision_at_5_mod(u, pred)
            results.append({'cust_no': u, 'precision_at_5': p_at_5})
            if p_at_5 > 0:
                p += self.precision_at_5(u, pred)
        results = pd.DataFrame(results)
        return p/count, results
    
    def precision_at_5_mod(self, user, pred):
        try:
            y_true = self.ans[user]
            tp = len(set(y_true) & set(pred))
            return tp/5
        except:
            return -1

def get_esun_train_interactions(dataset_name:str = '2018-12-31_6', overwrite=False, rating=False):
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
        if rating:
            interval = (inter_df['timestamp'].max() - inter_df['timestamp'].min()) / 4
            inter_df['rating'] = ((inter_df['timestamp'] - inter_df['timestamp'].min()) / interval).astype('int') + 1
            inter_df.sort_values('rating', ascending=False, inplace=True)
            inter_df.drop_duplicates(['cust_no', 'wm_prod_code'], keep='first', inplace=True)
        else:
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
# %%
if __name__ == '__main__':
    df = get_esun_train_interactions(rating=True, overwrite=True)
    print(df)

#%%