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

torch.manual_seed(params["SEED"])  # pytorch random seed
np.random.seed(params["SEED"])  # numpy random seed
torch.backends.cudnn.deterministic = True  

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

print(params)

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

    if params["MODE"] == "train_set":
        train_dataset = dataset["train"]
        path_train = "../datalength/train_info.csv"
        df_train = pd.read_csv(path_train)
        index_train = get_index(df_train, params)
        train_loader = get_loader(train_dataset, index_train, params)
        console.log(f"[Loading Data]...PASS!\n")
        return train_loader
        
    elif params["MODE"] == "test_set":
        test_dataset = dataset["test"]
        path_test = "../datalength/test_info.csv"
        df_test = pd.read_csv(path_test)  
        index_test = get_index(df_test, params)
        test_loader = get_loader(test_dataset, index_test, params)
        console.log(f"[Loading Data]...PASS!\n")
        return test_loader
    else:
        return 
 
def get_index(df, params):   
    if params["RESTRICTION"] == True:
        print("LEN RESTRICTION")
        # less than n input tokens

        cond1 = df["doc len"] >= 485
        cond2 = df["doc len"] <= 512
        cond3 = df["sum len"] <= 36

#         testcond1 = df_test["doc len"] >= 485
#         testcond2 = df_test["doc len"] <= 512
#         testcond3 = df_test["sum len"] <= 36

        index = df['index'][cond1 & cond2 & cond3]
        index = np.random.permutation(index)
#         index_test = df_test['index'][testcond1 & testcond2 & testcond3]
#         print(len(index_train))
#         print(len(index_test))
        return index
    else:
        print("NO LEN RESTRICTION")
        index = df['index']
        index = np.random.permutation(index)
        return index

#     console.print(f"Filtered TRAIN Dataset: {len(train_dataset[index_train]['id'])}")
#     console.print(f"Filtered TEST Dataset: {len(test_dataset[index_test]['id'])}\n")

def get_loader(dataset, index, params):
    tokenizer = T5Tokenizer.from_pretrained(params["MODEL"])
    
    data_set = Dataset(
        dataset[index],
        tokenizer,
        params["MODEL"],
        params["MAX_SOURCE_TEXT_LENGTH"],
        params["MAX_TARGET_TEXT_LENGTH"],
        params["SOURCE TEXT"],
        params["TARGET TEXT"],
        params["METHOD"],
    )  
    loader_params = {
        "batch_size": params["BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }  
    
    loader = DataLoader(data_set, **loader_params)
    print(f"""LOADER {params["MODE"]}: """, len(loader))
    return loader
    
def preprocess(loader, params):
    results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] } 
    for _, data in enumerate(loader, 0):
        results['Sample ids'].extend(data['ids'])
        results['Document'].extend(data['source_text'])
        results['Shortened Document'].extend(data['shortened_source_text'])
        results['Summary'].extend(data["target_text"])
        results['Document length'].extend(data['source_len'].tolist())
        print("STEP: ", _,"/",len(loader))
        final_df = pd.DataFrame(results)
        if not os.path.exists(f"""preprocessed_text/{params["METHOD"]}_test/quantity_{params["SHORTENING QUANTITY"]}/{params["MODE"]}"""):
            os.makedirs(f"""preprocessed_text/{params["METHOD"]}_test/quantity_{params["SHORTENING QUANTITY"]}/{params["MODE"]}""")
        final_df.to_csv(f"""preprocessed_text/{params["METHOD"]}_test/quantity_{params["SHORTENING QUANTITY"]}/{params["MODE"]}/{params["METHOD"]}_quantity{params["SHORTENING QUANTITY"]}_step{_}.csv""")
        print("SAVE TO CSV FINISHED")
        results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] } 
    


if __name__ == "__main__":  
    loader = preparedata(data, params)
    preprocess(loader, params)
    