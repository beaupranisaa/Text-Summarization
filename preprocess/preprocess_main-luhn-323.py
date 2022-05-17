# Importing libraries
import sys         
sys.path.append('/home/pranisaa/working_dir/Text-Summarization')

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pickle
from preprocess_utils import checker
from torch import cuda

from datasets import load_dataset
from preprocess_config import *
from preprocess_dataset import Dataset

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console
from IPython.display import clear_output
import math

from transformers import T5Tokenizer

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

def preparedata(data, config):
    if data == 'cnn_dailymail':
        dataset = load_dataset(data, '3.0.0')
        config['source_text'] = "article"
        config['target_text'] = "highlights"
    elif data == "xsum":
        dataset = load_dataset(data)
        config['source_text']  = "document"
        config['target_text'] = "summary"
    else:
        raise ValueError("Undefined dataset")

    train_dataset, val_dataset, test_dataset = get_dataset(dataset, "train", config), get_dataset(dataset, "validation", config), get_dataset(dataset, "test", config)
    
    training_loader, val_loader,  test_loader = get_loader(train_dataset, config), get_loader(val_dataset, config), get_loader(test_dataset, config)
    
    return training_loader, val_loader,  test_loader

def get_dataset(dataset, mode, config):      
    data = dataset[mode]
    path =  f"../datalength/{mode}_info.csv"
    df = pd.read_csv(path)

    if config['max_source_length'] == None:
        print("NO LEN RESTRICTION")
        index = df['index']
        
    else:
        print("LEN RESTRICTION")
        # less than n input tokens
        if config['orig_source_length'] == 512:
            cond1 = df["doc len"] >= math.floor(0.95*config['orig_source_length']-1) #485, 972
        elif config['orig_source_length'] == 1024:
            cond1 = df["doc len"] >= math.floor(0.95*config['orig_source_length'])
        else:
            raise ValueError("undefined...")
        cond2 = df["doc len"] <= config['orig_source_length'] #512, 1024
        cond3 = df["sum len"] <= config['max_target_length']

        index = df['index'][cond1 & cond2 & cond3]

    # Shuffle and take only 1000 samples
    index = np.random.permutation(index)[3584:]
    
    print(f"Filtered {mode} Dataset: {len(data[index]['id'])}")
    
    tokenizer = T5Tokenizer.from_pretrained(config['model'])

    # Creating the Training and Validation dataset for further creation of Dataloader
    data_set = Dataset(
        data[index],
        tokenizer,
        config['model'],
        config['max_source_length'],
        config['max_target_length'],
        config['source_text'],
        config['target_text'],
        config['method'],
    )
    
    return data_set

def get_loader(dataset, config): 
    loader_params = {
        "batch_size": config['batch_size'],
        "shuffle": False,
        "num_workers": 0,
    }   
    loader = DataLoader(dataset, **loader_params)
    
    print(f"Loader Length: {len(loader)}")
    return loader


def process(loader, config, mode):
    results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Shortened Document length": []} 
    path = f"""preprocessed_text/{config['orig_source_length']}_{config['method']}/quantity_{config['max_source_length']}/{mode}_fix/"""
    if not os.path.exists(path):
        os.makedirs(path)        
    for _, data in enumerate(loader, 0):
        _ = _ + 28
        results['Sample ids'].extend(data['ids'])
        results['Document'].extend(data['source_text'])
        results['Shortened Document'].extend(data['shortened_source_text'])
        results['Summary'].extend(data["target_text"])
        results['Document length'].extend(data['source_len'].tolist())
        results["Shortened Document length"].extend(data['source_text_short_len'].tolist())
        
        print("STEP: ", _,"/",len(loader))
        final_df = pd.DataFrame(results)
        final_df.to_csv(os.path.join(path, f"""step{_}.csv"""""))   
        print("SAVE TO CSV FINISHED")
        results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Shortened Document length": []}                  

config = dict(
    model = "t5-small",
    data = "xsum",    
    batch_size = 128,
    orig_source_length = 512,
    max_source_length = 323, # 512, 373, 323, 273
    max_target_length = 36,
    method = "luhn",
    seed = 42,
)
            
if __name__ == "__main__": 
    
    torch.manual_seed(config['seed'])  # pytorch random seed
    np.random.seed(config['seed'])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    train_loader, val_loader, test_loader = preparedata(data, config)
    
#     process(test_loader, config, "test_set")
#     process(val_loader, config, "val_set")
    process(train_loader, config, "train_set")
     