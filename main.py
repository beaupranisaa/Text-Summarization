# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import pickle
from train import Trainer

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
    "TRAIN_BATCH_SIZE": 16,  # training batch size
    "VALID_BATCH_SIZE": 16,  # validation batch size
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 2e-05,  # learning rate default betas=(0.9, 0.999), eps=1e-08
    "SCHEDULER": "linear",
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}

# path_train = "train_data_length_info.csv"
# df_train = pd.read_csv(path_train)
# index_train = df_train['index'][df_train["length"] < 512]

Trainer(
    dataset=dataset,
    source_text=source_text, 
    target_text=target_text,
    model_params=model_params,
    output_dir=f"""./outputs_{model_params["MODEL"]}_{data}/""",
    device = device,
    mask = False,
    to_mask_list = None
)
