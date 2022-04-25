# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console
from IPython.display import clear_output
import numpy as np
import os
import re

console = Console(record=True)

def display_dataset(ds):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(ds.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)
    
def checker(model_params):
    assert model_params["SHORTENING QUANTITY"] in [0, 25, 35, 45, 'all']
    assert model_params["MAX_SOURCE_TEXT_LENGTH"] in [512, 373, 323, 273, '-']
    assert model_params["METHOD"] in ["full-text", "head-only", "tail-only", "luhn", "lsa", "textrank", "stopwords"]+["head+tail_ratio{:.1f}".format(i) for i in np.arange(0.0, 1.0, 0.1)]

    if model_params["METHOD"] == "full-text":
        assert model_params["SHORTENING QUANTITY"] == 0
        assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 512
    
    if model_params["SHORTENING QUANTITY"] == 25:
        assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 373
    elif model_params["SHORTENING QUANTITY"] == 35:
        assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 323
    elif model_params["SHORTENING QUANTITY"] == 45:
        assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 273
    elif model_params["SHORTENING QUANTITY"] == "all":
        assert model_params["MAX_SOURCE_TEXT_LENGTH"] == '-'
    else:
        pass

def get_last_checkpoint(path):
    if not os.path.exists(path):
        raise ValueError("No checkpoints to resume, please start training to create checkpoints...")
    else:
        content = os.listdir(path)
        checkpoints = [path for path in content]
        last_checkpoints = max([int(re.findall(r"\d*\d", cp)[0]) for cp in checkpoints if len(re.findall(r"\d*\d", cp)) != 0])   
        
        if last_checkpoints == 0:
            raise ValueError("No checkpoints to resume, please start training to create checkpoints...")
        else:
            if len(checkpoints) == 0:
                return
            return os.path.join(path, f"epoch{last_checkpoints-1}"), last_checkpoints-1
