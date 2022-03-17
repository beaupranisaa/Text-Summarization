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

from config import *

data = data

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
model_params = model_params

# let's define path
path = path

Trainer(
    dataset=dataset,
    source_text=source_text, 
    target_text=target_text,
    model_params=model_params,
    output_dir= path, #f"""./model/{model_params["MODEL"]}_{data}_nolenrestriction/""",
    device = device,
    len_restriction = len_restriction,
    mask = mask,
    to_mask_list = to_mask_list
)
