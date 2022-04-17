# from transformers import T5Tokenizer,
from torch.utils.data import Dataset
import torch
import pandas as pd
from tokenizers import decoders

class Dataset(Dataset):
    """u
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, model_name, source_len, target_len, source_text, target_text, to_mask_list = None,  mask = False,#train = True,
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
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.mask = mask
        self.to_mask_list = to_mask_list
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
            max_length = 512, #self.source_len
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
        source_len = self.tokenizer.batch_encode_plus(
            [source_text],
            return_tensors="pt",
        )
        
        target_len = self.tokenizer.batch_encode_plus(
            [target_text],
            return_tensors="pt",
        )
        
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()        
        
        # get end token
        end_eos = int(torch.where(source_ids == 1)[0])
        
        # tail-only
#         source_ids_short = source_ids[end_eos-self.source_len+1:end_eos+1]
#         source_ids = source_ids_short
#         source_ids[0], source_ids[1] = 21603, 10  # 21603, 10 = summarize, :      
#         source_mask = source_mask[end_eos-self.source_len+1:end_eos+1]
        
        # Head+tail
        head_ids, head_mask = source_ids[0:self.source_len//2+1], source_mask[0:self.source_len//2+1]
        tail_ids, tail_mask= source_ids[end_eos-(self.source_len//2)+1:end_eos+1], source_mask[end_eos-(self.source_len//2)+1:end_eos+1]
        source_ids_short = torch.cat((head_ids, tail_ids),0)
        source_ids = source_ids_short
        source_mask = torch.cat((head_mask, tail_mask),0)

        # padding
        diff = 512 - len(source_ids)
        pad = torch.zeros(diff)
        source_ids = torch.cat((source_ids, pad), 0)
        source_mask = torch.cat((source_mask, pad), 0)

        tokenizer = decoders.WordPiece()
        source_ids_short = self.tokenizer.decode(source_ids_short)

        return {
            "source_text": source_text,
            "shortened_source_text": source_ids_short,
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "source_len": len(source_len["input_ids"].squeeze()),
            "target_text": target_text,
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
            "target_len": len(target_len["input_ids"].squeeze()),
            "ids": ids,
        }
    
    def add_prefix(self, examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples]
        return inputs
    
    def count_words(self):
        pass

class EvalDataset(Dataset):
    
    def __init__(self, path):
        #read csv file and load row data into variable
        file_out = df = pd.read_csv(path)
        self.ref = file_out['Reference summary']
        self.hyp = file_out['Generated summary']
        
    def __len__(self):
        return len(self.hyp)
    
    def __getitem__(self, idx):
        return self.ref[idx], self.hyp[idx]