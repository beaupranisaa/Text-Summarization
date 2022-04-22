import numpy as np
import pandas as pd
from preprocess_dataset import Dataset
import torch
from torch.utils.data import DataLoader

from transformers import T5Tokenizer

def checker(model_params):
    assert model_params["SHORTENING QUANTITY"] in [0, 25, 35, 45]
    assert model_params["MAX_SOURCE_TEXT_LENGTH"] in [512, 373, 323, 273]
    assert model_params["METHOD"] in ["luhn", "textrank", "lsa"]

    if model_params["SHORTENING QUANTITY"] == 25:
        assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 373
    elif model_params["SHORTENING QUANTITY"] == 35:
        assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 323
    elif model_params["SHORTENING QUANTITY"] == 45:
        assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 273
    else:
        pass
