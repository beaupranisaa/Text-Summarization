# Importing libraries
import sys         
sys.path.append('/home/pranisaa/working_dir/Text-sum-test')

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

from transformers import T5Tokenizer

console = Console(record=True)

training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

console = Console(record=True)

training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

torch.manual_seed(params["SEED"])  # pytorch random seed
np.random.seed(params["SEED"])  # numpy random seed
torch.backends.cudnn.deterministic = True

console.log(f"[Checking configuration]...\n")
checker(params)
console.log(f"[Checking configuration]...PASS!\n")


def preparedata(data, params):
    
    if data == 'cnn_dailymail':
        dataset = load_dataset(data, '3.0.0')
        params["SOURCE TEXT"] = "article"
        params["TARGET TEXT"] = "highlights"
    elif data == "xsum":
        dataset = load_dataset(data)
        params["SOURCE TEXT"] = "document"
        params["TARGET TEXT"] = "summary"
    else:
        raise ValueError("Undefined dataset")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    path_train = "../datalength/train_info.csv"
    df_train = pd.read_csv(path_train)

    path_test = "../datalength/test_info.csv"
    df_test = pd.read_csv(path_test)    

    if params["RESTRICTION"] == True:
        print("LEN RESTRICTION")
        # less than n input tokens

        traincond1 = df_train["doc len"] >= 485
        traincond2 = df_train["doc len"] <= 512
        traincond3 = df_train["sum len"] <= 36

        testcond1 = df_test["doc len"] >= 485
        testcond2 = df_test["doc len"] <= 512
        testcond3 = df_test["sum len"] <= 36

        index_train = df_train['index'][traincond1 & traincond2 & traincond3]
        index_test = df_test['index'][testcond1 & testcond2 & testcond3]
        print(len(index_train))
        print(len(index_test))

    else:
        print("NO LEN RESTRICTION")
        index_train = df_train['index']
        index_test = df_test['index']

    # Shuffle and take only 1000 samples
    index_train = np.random.permutation(index_train)
    index_test = np.random.permutation(index_test)
    #     print(index_train)
    
    console.print(f"Filtered TRAIN Dataset: {len(train_dataset[index_train]['id'])}")
    console.print(f"Filtered TEST Dataset: {len(test_dataset[index_test]['id'])}\n")

    del dataset

    tokenizer = T5Tokenizer.from_pretrained(params["MODEL"])

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = Dataset(
        train_dataset[index_train],
        tokenizer,
        params["MODEL"],
        params["MAX_SOURCE_TEXT_LENGTH"],
        params["MAX_TARGET_TEXT_LENGTH"],
        params["SOURCE TEXT"],
        params["TARGET TEXT"],
        params["METHOD"],
    )
    test_set = Dataset(
        test_dataset[index_test],
        tokenizer,
        params["MODEL"],
        params["MAX_SOURCE_TEXT_LENGTH"],
        params["MAX_TARGET_TEXT_LENGTH"],
        params["SOURCE TEXT"],
        params["TARGET TEXT"],
        params["METHOD"]
    )

    del train_dataset, test_dataset

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

# results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] } 

def process(loader, params, mode):
    results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] } 
    for _, data in enumerate(loader, 0):
        results['Sample ids'].extend(data['ids'])
        results['Document'].extend(data['source_text'])
        results['Shortened Document'].extend(data['shortened_source_text'])
        results['Summary'].extend(data["target_text"])
        results['Document length'].extend(data['source_len'].tolist())
        print("STEP: ", _,"/",len(loader))
        final_df = pd.DataFrame(results)
        if not os.path.exists(f"""preprocessed_text/{params["METHOD"]}_test/quantity_{params["SHORTENING QUANTITY"]}/{mode}"""):
            os.makedirs(f"""preprocessed_text/{params["METHOD"]}_test/quantity_{params["SHORTENING QUANTITY"]}/{mode}""")
        final_df.to_csv(f"""preprocessed_text/{params["METHOD"]}_test/quantity_{params["SHORTENING QUANTITY"]}/{mode}/{params["METHOD"]}_quantity{params["SHORTENING QUANTITY"]}_step{_}.csv""")
        print("SAVE TO CSV FINISHED")
        results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] } 

if __name__ == "__main__":  
    train_loader, test_loader = preparedata(data, params)
    process(test_loader, params, "test_set")
    process(train_loader, params, "train_set")
    