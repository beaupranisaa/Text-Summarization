import math
import re
import torch

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer

from transformers import T5Tokenizer

class Strategy:
    def __init__(self, source_text, source_ids, source_mask, source_len, max_source_len):
        self.source_text = source_text
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.source_len = source_len
        self.end_eos = int(torch.where(self.source_ids == 1)[0])
        self.max_source_len = max_source_len
    
    def shorten(self, method):
        if 'combo' in method:
            if "stopwords" in method and "head+tail" in method:
                head_ratio = float(re.findall(r"\d*\.\d", method)[0])
                source_ids, source_ids_short, source_mask = self._get_headtail(self.source_ids, self.source_mask, self.end_eos, self.max_source_len, head_ratio)
            elif "stopwords" in method and "luhn" in method or "textrank" in method:
                source_ids, source_ids_short, source_mask = self._get_head(self.source_ids, self.source_mask, self.max_source_len) 
            else: 
                raise ValueError("Undefined shortening combo strategy...") 
        else:
            if "full-text" in method:
                source_ids, source_ids_short, source_mask = self._get_fulltext(self.source_ids, self.source_mask, self.source_len)
            elif "head-only" in method:
                source_ids, source_ids_short, source_mask = self._get_head(self.source_ids, self.source_mask, self.max_source_len)         
            elif "tail-only" in method:
                source_ids, source_ids_short, source_mask = self._get_tail(self.source_ids, self.source_mask, self.end_eos, self.max_source_len)
            elif "head+tail" in method:
                head_ratio = float(re.findall(r"\d*\.\d", method)[0])
                source_ids, source_ids_short, source_mask = self._get_headtail(self.source_ids, self.source_mask, self.end_eos, self.max_source_len, head_ratio)
            elif "luhn" in method:
                source_ids, source_ids_short, source_mask = self._get_head(self.source_ids, self.source_mask, self.max_source_len)      
            elif "textrank" in method:
                source_ids, source_ids_short, source_mask = self._get_head(self.source_ids, self.source_mask, self.max_source_len) 
            elif "lsa" in method:
                source_ids, source_ids_short, source_mask = self._get_head(self.source_ids, self.source_mask, self.max_source_len) 
            elif "bertbased" in method:
                source_ids, source_ids_short, source_mask = self._get_head(self.source_ids, self.source_mask, self.max_source_len) 
            # force to get stop and then head+tail
            elif "stopwords" in method:
                source_ids, source_ids_short, source_mask = self._get_fulltext(self.source_ids, self.source_mask, self.source_len)
#                 head_ratio = float('0.5')
#                 source_ids, source_ids_short, source_mask = self._get_headtail(self.source_ids, self.source_mask, self.end_eos, self.max_source_len, head_ratio)

            else:
                raise ValueError("Undefined shortening strategy...") 

        return source_ids, source_ids_short, source_mask
        
    def _get_fulltext(self, source_ids, source_mask, source_len):
        source_ids_short = source_ids[0:source_len]
        source_ids = source_ids_short
        source_mask = source_mask[0:source_len]
        source_ids, source_mask = self._get_padding(source_ids, source_mask)
        return source_ids, source_ids_short, source_mask  
    
    def _get_head(self, source_ids, source_mask, max_source_len):
        source_ids_short = source_ids[0:max_source_len]   
        source_ids = source_ids_short
        source_ids[-1] = 1
        source_mask = source_mask[0:max_source_len]
        source_ids, source_mask = self._get_padding(source_ids, source_mask)
        return source_ids, source_ids_short, source_mask        
    
    def _get_tail(self, source_ids, source_mask, end_eos, max_source_len):
        source_ids_short = source_ids[end_eos-max_source_len+1:end_eos+1]
        source_ids = source_ids_short
        source_ids[0], source_ids[1] = 21603, 10  # 21603, 10 = summarize, :      
        source_mask = source_mask[end_eos-max_source_len+1:end_eos+1]
        source_ids, source_mask = self._get_padding(source_ids, source_mask)
        return source_ids, source_ids_short, source_mask
    
    def _get_headtail(self, source_ids, source_mask, end_eos, max_source_len, head_ratio):
        head_ids, head_mask = source_ids[0:math.floor(head_ratio*max_source_len)+1], source_mask[0:math.floor(head_ratio*max_source_len)+1]
        tail_ids, tail_mask= source_ids[end_eos - math.floor((1-head_ratio)*max_source_len)+1:end_eos+1], source_mask[end_eos - math.floor((1-head_ratio)*max_source_len)+1:end_eos+1]
        source_ids_short = torch.cat((head_ids, tail_ids),0)
        source_ids = source_ids_short
        source_mask = torch.cat((head_mask, tail_mask),0)
        source_ids, source_mask = self._get_padding(source_ids, source_mask)
        return source_ids, source_ids_short, source_mask
    
    def _get_padding(self, source_ids, source_mask):
        diff = 512 - len(source_ids)
        pad = torch.zeros(diff)
        source_ids = torch.cat((source_ids, pad), 0)
        source_mask = torch.cat((source_mask, pad), 0)
        return source_ids, source_mask
    