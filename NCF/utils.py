import zipfile
from torch.utils.data import Dataset
import torch
import os
import pandas  as pd
import numpy as np
from zipfile import ZipFile
import requests
import sklearn
import random

class LoadDataset():
    def __init__(self,root: str,file_category: str = '1m') -> None:
        self.root = root
        # self.download = download
        self.file_category_url = file_category
        if self.file_category_url =='1m':
            self.file_url = 'ml-' + self.file_category_url
            self.fname = os.path.join(self.root, self.file_url, 'ratings.dat')
        if self.file_category_url == 'Beauty':
            self.file_url =  self.file_category_url
            self.fname = os.path.join(self.root, self.file_url, 'Beauty.inter')
        if self.file_category_url == 'yelp':
            self.file_url =  self.file_category_url
            self.fname = os.path.join(self.root, self.file_url, 'Yelp.inter')
        self.df = self._read_ratings_csv()
    def _read_ratings_csv(self) -> pd.DataFrame:
        '''
        at first, check if file exists. if it doesn't then call _download().
        it will read ratings.csv, and transform to dataframe.
        it will drop columns=['timestamp'].
        :return:
        '''
        print("Reading file")
        if self.fname=='../dataset/ml-1m/ratings.dat':
            df = pd.read_csv(self.fname, sep="::", header=None,names=['userId', 'itemId', 'ratings', 'timestamp'])
            df = df.drop(columns=['timestamp'])
            # print(df.dtypes)
        if self.fname=='../dataset/Beauty/Beauty.inter':
            df = pd.read_csv(self.fname, sep="\t", header=None,names=['userId', 'itemId', 'timestamp'])
            df = df.drop([0])
            df = df.reindex()
            df = pd.DataFrame(df,dtype=int)
            df = df.drop(columns=['timestamp'])
        if self.fname=='../dataset/yelp/Yelp.inter':
            df = pd.read_csv(self.fname, sep="\t", header=None,names=['userId', 'itemId'])
            df = df.drop([0])
            df = df.reindex()
            df = pd.DataFrame(df,dtype=int)
        
        print("Reading Complete!")
        return df

    def split_train_test(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        '''
        pick each unique userid row, and add to the testset, delete from trainset.
        :return: (pd.DataFrame,pd.DataFrame,pd.DataFrame)
        '''
        train_dataframe = self.df.sample(frac=0.7,random_state=1)
        test_dataframe = self.df.drop(train_dataframe.index)
        train_dataframe.loc[:, 'rating'] = 1
        test_dataframe.loc[:, 'rating'] = 1

        print(f"len(total): {len(self.df)}, len(train): {len(train_dataframe)}, len(test): {len(test_dataframe)}")
        return self.df, train_dataframe, test_dataframe,


class DatasetSplit(Dataset):
    def __init__(self,df: pd.DataFrame,total_df: pd.DataFrame,ng_ratio:int)->None:
        '''
        :param root: dir for download and train,test.
        :param df: parsed dataframe
        :param file_size: large of small. if size if large then it will load(download) 20M dataset. if small then, it will load(download) 100K dataset.
        :param download: if true, it will down load from url.
        '''
        super(DatasetSplit, self).__init__()
        self.df = df
        self.total_df = total_df
        self.ng_ratio = ng_ratio

        # self._data_label_split()
        self.users, self.items, self.labels = self._negative_sampling()

    def __len__(self) -> int:
        '''
        get lenght of data
        :return: len(data)
        '''
        return len(self.users)


    def __getitem__(self, index):
        '''
        transform userId[index], item[inedx] to Tensor.
        and return to Datalaoder object.
        :param index: idex for dataset.
        :return: user,item,rating
        '''
        return self.users[index], self.items[index], self.labels[index]


    def _negative_sampling(self) :
        '''
        sampling one positive feedback per #(ng ratio) negative feedback
        :return: list of user, list of item,list of target
        '''
        df = self.df
        total_df = self.total_df
        users, items, labels = [], [], []
        user_item_set = set(zip(df['userId'], df['itemId']))
        total_user_item_set = set(zip(total_df['userId'],total_df['itemId']))
        all_movieIds = total_df['itemId'].unique()
        # negative feedback dataset ratio
        negative_ratio = self.ng_ratio
        for u, i in user_item_set:
            # positive instance
            users.append(u)
            items.append(i)
            labels.append(1.0)
            #visited check
            visited=[]
            visited.append(i)
            # negative instance
            for i in range(negative_ratio):
                # first item random choice
                negative_item = np.random.choice(all_movieIds)
                # check if item and user has interaction, if true then set new value from random
                while (u, negative_item) in total_user_item_set or negative_item in visited :
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                visited.append(negative_item)
                labels.append(0.0)
        print(f"negative sampled data: {len(labels)}")
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)




