import os
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import pickle

# Load dataset
from datasets import load_dataset

model = "t5-small"
data = 'xsum'

if data == 'cnn_dailymail':
    dataset = load_dataset(data, '3.0.0')
    source_text = "article"
    target_text = "highlights"
elif data == "xsum":
    dataset = load_dataset(data)
    source_text = "document"
    target_text = "summary"
else:
    raise ValueError("Undefined dataset")
    
EPOCH = 50

for epoch in range(1,EPOCH):
    
    path_gen = f'{model}_{data}_nomask/outputs_{model}_{data}/results_gen/predictions_t5-small_epoch{epoch}.csv'
    path_eval = f'{model}_{data}_nomask/outputs_{model}_{data}/results_eval/predictions_t5-small_epoch{epoch}.csv'
    df_gen = pd.read_csv(path_gen)
    df_eval = pd.read_csv(path_eval)
    
    columns_gen = []
    old_columns_gen = list(df_gen.columns)
    columns_gen.append("sample id")
    columns_gen.extend(old_columns_gen)
    
    columns_eval = []
    old_columns_eval = list(df_eval.columns)
    columns_eval.append("sample id")
    columns_eval.extend(old_columns_eval)
    
    df_gen["sample id"] = dataset['validation']['id']
    df_eval["sample id"] = dataset['validation']['id']
    
    df_gen = df_gen.reindex(columns=columns_gen)
    df_gen = df_gen.drop(columns=['Unnamed: 0'])
    
    df_eval = df_eval.reindex(columns=columns_eval)
    df_eval = df_eval.drop(columns=['Unnamed: 0'])
    
    df_gen.to_csv(path_gen, index=False,header=True)
    df_eval.to_csv(path_eval, index=False,header=True)
    print(f"epoch {epoch} DONE!")