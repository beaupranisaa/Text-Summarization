import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import pickle
from train import Trainer
from transformers import T5Tokenizer

import os
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print("configured device: ", device)

# Load dataset
from datasets import load_dataset

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

# let's define model parameters specific to BART
model_params = {
    "MODEL": "t5-small",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 1024,  # training batch size
    "VALID_BATCH_SIZE": 1024,  # validation batch size
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 2e-05,  # learning rate default betas=(0.9, 0.999), eps=1e-08
    "SCHEDULER": "linear",
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}

from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import random
class Dataset(Dataset):
    """u
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, data, tokenizer, model_name, source_len, target_len, source_text, target_text, to_mask_list = None,  mask = False,
    ):
        """
        Initializes a Dataset class

        Args:yo
            data : Input data
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_len = source_len
        self.summ_len = target_len
        self.mask = mask
        self.to_mask_list = to_mask_list
        
        self.source_text = self.data[source_text]
#         self.source_text_len = len(self.data[source_text])
        if "t5" in model_name:
            self.source_text = self.add_prefix(self.source_text)
        
        self.target_text = self.data[target_text]
        
        self.ids = self.data['id']
        
        self.ids_check = [i for i in range(len(self.source_text)-1)]

    def __len__(self):
        """returns the length of data"""

        return len(self.target_text)

    def __getitem__(self, index):
#         print("INDEX: ", index)
        """return the input ids, attention masks and target ids"""
    
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        ids = self.ids[index]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source_len = self.tokenizer.batch_encode_plus(
            [source_text],
            return_tensors="pt",
        )

        len_source = len(source_len["input_ids"].squeeze())
        if self.mask == True:
            source_text = [x for x in source_text.split() if x not in self.to_mask_list ]
            source_text = " ".join(source_text)
        
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )        
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
            "source_len": len(source_len["input_ids"].squeeze()),
            "ids": ids,
            "index" : index
            }            

    def add_prefix(self, examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples]
        return inputs
    
    def compute_tfidf(self):
        
        text_tfidf = self._prepare_data_tfidf()
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=False) #use_idf bool, default=True (to highlight by comparison) Enable inverse-document-frequency reweighting
        x = tfidf_vectorizer.fit_transform(text_tfidf)
        tfidfcounts = pd.DataFrame(x.toarray(),index = self.source_text,  columns = tfidf_vectorizer.get_feature_names())
        return tfidfcounts
    
    def _prepare_data_tfidf(self):
        source_text_lst = []
        for i, source in enumerate(self.source_text):
            source = str(source)
            source = " ".join(source.split())
            source_text_lst.append(source) 
        source_text = pd.Series(source_text_lst)
        return source_text
    
tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
training_set = Dataset(
    train_dataset,
    tokenizer,
    model_params["MODEL"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
    source_text,
    target_text,
    mask = False,
    to_mask_list = None,
#         train = True,
)
val_set = Dataset(
    val_dataset,
    tokenizer,
    model_params["MODEL"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
    source_text,
    target_text,
#         train = False,
)
train_params = {
    "batch_size": model_params["TRAIN_BATCH_SIZE"],
    "shuffle": False,
    "num_workers": 0,
}

val_params = {
    "batch_size": model_params["VALID_BATCH_SIZE"],
    "shuffle": False,
    "num_workers": 0,
}

# Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(val_set, **val_params)

sample_ids = []
sample_index = []
sample_len = []
for _, data in enumerate(val_loader, 0):
    print("STEP: ", _,"/", len(val_loader))
    sample_len.extend(data["source_len"].tolist())
    sample_ids.extend(data['ids'])
    sample_index.extend(data['index'].tolist())


df = pd.DataFrame({"ids": sample_ids, 
                   "index": sample_index,
                   "length":sample_len
                  })

df.to_csv("val_data_length_info.csv")