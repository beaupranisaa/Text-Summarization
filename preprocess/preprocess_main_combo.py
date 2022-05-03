# Importing libraries
import sys         
sys.path.append('/home/pranisaa/working_dir/Text-Summarization')

import os
import pandas as pd
from torch.utils.data import DataLoader
import torch
import numpy as np

import pickle
from preprocess_utils import checker

from datasets import load_dataset
from preprocess_config import *
from preprocess_dataset import Dataset

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console
from IPython.display import clear_output

from transformers import T5Tokenizer

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

torch.manual_seed(params["SEED"])  # pytorch random seed
np.random.seed(params["SEED"])  # numpy random seed
torch.backends.cudnn.deterministic = True

def preparedata(data, params):

    preprocessed_data_train = pd.read_csv(f"""preprocessed_text/{params["METHOD1"]}/quantity_{params["SHORTENING QUANTITY1"]}/train_set.csv""")
    preprocessed_data_test = pd.read_csv(f"""preprocessed_text/{params["METHOD1"]}/quantity_{params["SHORTENING QUANTITY1"]}/test_set.csv""")
    
    preprocessed_data_train.rename(columns = {'Sample ids':'id'}, inplace = True)
    preprocessed_data_test.rename(columns = {'Sample ids':'id'}, inplace = True)
    
    tokenizer = T5Tokenizer.from_pretrained(params["MODEL"])
    
    training_set = Dataset(
        preprocessed_data_train,
        tokenizer,
        params["MODEL"],
        params["MAX_SOURCE_TEXT_LENGTH"],
        params["MAX_TARGET_TEXT_LENGTH"],
        "Shortened Document",
        "Summary",
        params["METHOD2"],
        "combo",
    )
    test_set = Dataset(
        preprocessed_data_test,
        tokenizer,
        params["MODEL"],
        params["MAX_SOURCE_TEXT_LENGTH"],
        params["MAX_TARGET_TEXT_LENGTH"],
        "Shortened Document",
        "Summary",
        params["METHOD2"],
        "combo"
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": params["BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    test_params = {
        "batch_size": params["BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    test_loader = DataLoader(test_set, **test_params)

    print("TRAIN LOADER: ", len(training_loader))
    print("VAL LOADER: ", len(test_loader))

    del train_params, test_params   
    return training_loader, test_loader

def process(loader, params, mode):
    if params["METHOD2"] in ["stopwords"]:
        results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Removed words": []} 
    else:
        results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [], "Shortened Document len": [] } 
    for _, data in enumerate(loader, 0):
        results['Sample ids'].extend(data['ids'].tolist())
        results['Document'].extend(data['source_text'])
        results['Shortened Document'].extend(data['shortened_source_text'])
        results['Summary'].extend(data["target_text"])
        results['Document length'].extend(data['source_len'].tolist())
        if params["METHOD2"] == "stopwords":
            results["Removed words"].extend(data['n_stopwords'].tolist())
        else:
            results["Shortened Document len"].extend(data['source_text_short_len'].tolist())
        print("STEP: ", _,"/",len(loader))
        final_df = pd.DataFrame(results)
        path = f"""preprocessed_text/combo_{params["METHOD1"]}_{params["SHORTENING QUANTITY1"]}_{params["METHOD2"]}/quantity_{params["SHORTENING QUANTITY2"]}/{mode}"""
        if not os.path.exists(path):
            os.makedirs(path)
        final_df.to_csv(os.path.join(path, f"""combo_{params["METHOD1"]}_{params["SHORTENING QUANTITY1"]}_{params["METHOD2"]}_{params["SHORTENING QUANTITY2"]}step{_}.csv"""))
        print("SAVE TO CSV FINISHED")
        if params["METHOD2"] == "stopwords":
            results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Removed words": []} 
        else: results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [], "Shortened Document len": [] } 

if __name__ == "__main__":  
    train_loader, test_loader = preparedata(data, params_combo)
    process(test_loader, params_combo, "test_set")
    process(train_loader, params_combo, "train_set")
    