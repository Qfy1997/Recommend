import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import DatasetSplit,LoadDataset
from model.MLP import MLP
from model.GMF import GMF
from model.NeuMF import NeuMF
from train import Train
from evaluation import metrics
import os
import numpy as np
import time
from parser import args

root_path='../dataset'

if __name__=='__main__':
    # dataset = LoadDataset(root=root_path,file_category='Beauty')
    dataset = LoadDataset(root=root_path,file_category='yelp')

    print(type(dataset))
    print(dataset.df)
    total_dataframe, train_dataframe, test_dataframe = dataset.split_train_test()
    train_set = DatasetSplit(df=train_dataframe,total_df=total_dataframe,ng_ratio=4)
    test_set = DatasetSplit(df=test_dataframe,total_df=total_dataframe,ng_ratio=2)
    print(train_set[0])
    print(train_set[1])
    print(train_set[2])
    