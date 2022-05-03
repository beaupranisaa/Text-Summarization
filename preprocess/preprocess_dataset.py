# from transformers import T5Tokenizer,
import math
from torch.utils.data import Dataset
import torch
import pandas as pd
from tokenizers import decoders
import re
import sys   
from preprocess_strategy import *
sys.path.append('/home/pranisaa/working_dir/Text-Summarization')

from config import *
import numpy as np

torch.manual_seed(model_params["SEED"])  # pytorch random seed
np.random.seed(model_params["SEED"])  # numpy random seed
torch.backends.cudnn.deterministic = True

class Dataset(Dataset):
    """u
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, model_name, max_source_len, target_len, source_text, target_text, method = "luhn", mode = None, 
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.data = dataframe
        self.max_source_len = max_source_len
        self.summ_len = target_len
        self.mask = mask
        self.to_mask_list = to_mask_list
        self.method = method
        self.mode = mode
        self.source_text = self.data[source_text]
        if "t5" in model_name:
            self.source_text = self.add_prefix(self.source_text)
        self.target_text = self.data[target_text]
        
        self.ids = self.data['id']

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        
        ids = self.ids[index]
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        
        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())
            
        if self.mask == True:
            source_text = [x for x in source_text.split() if x not in self.to_mask_list ]
            source_text = " ".join(source_text)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length = 512, #self.max_source_len
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        source_len = self.tokenizer.batch_encode_plus(
            [source_text],
            return_tensors="pt",
        )
        
        
        source_ids = source["input_ids"].squeeze()
    
        strategy = SentenceLevelStrategy(self.source_text[index], source_ids, source_len, self.max_source_len, self.mode)
        
        if "stopwords" in self.method:
            source_text_short, n_stopwords = strategy.shorten(self.method)
            
            return {
            "source_text": source_text,
            "shortened_source_text": source_text_short,
            "target_text": target_text,
            "source_len": len(source_len["input_ids"].squeeze()),
            "ids": ids,
            "n_stopwords" : n_stopwords,
            }
    
        
        source_text_short, source_text_short_len = strategy.shorten(self.method)

        return {
            "source_text": source_text,
            "shortened_source_text": source_text_short,
            "target_text": target_text,
            "source_len": len(source_len["input_ids"].squeeze()),
            "source_text_short_len": source_text_short_len,
            "ids": ids,
        }
    
    def add_prefix(self, examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples]
        return inputs
    
