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
        self.source_len = len(source_len["input_ids"].squeeze())
        self.end_eos = int(torch.where(self.source_ids == 1)[0])
        self.max_source_len = max_source_len
    
    def shorten(self, method):
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
            source_ids, source_ids_short, source_mask = self._get_luhn(self.source_text, self.source_ids, self.max_source_len)
        else:
            pass 
        
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
    
    def _get_luhn(self, source_text, source_ids, max_source_len):
        sentence_count = source_text.count("\n") + 1 # count number of sentences
#         print("orig: ", source_text)
#         print(source_text)
        token_count = torch.count_nonzero(source_ids)
        n_token_per_sentence = token_count/sentence_count
        n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
        parser = PlaintextParser.from_string(source_text[11:],Tokenizer("english")) #remove summarize: 
        summarizer = LuhnSummarizer()
        source_ids, source_ids_short, source_mask = self._get_summary(summarizer, parser, n_sum_sentence)
        return  source_ids, source_ids_short, source_mask
        
        
    def _get_padding(self, source_ids, source_mask):
        diff = 512 - len(source_ids)
        pad = torch.zeros(diff)
        source_ids = torch.cat((source_ids, pad), 0)
        source_mask = torch.cat((source_mask, pad), 0)
        return source_ids, source_mask
    
    def _get_summary(self, summarizer, parser, n_sum_sentence):    
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        temp = []
#         print("HI: ",n_sum_sentence)
        for i in range(n_sum_sentence-1, n_sum_sentence +2):
            summary = summarizer(parser.document,i)
            full_summary = ' '.join([sentence._text for sentence in summary])
            full_summary = "summarize: " + full_summary
            temp.append(full_summary)
        
        temp_source = tokenizer.batch_encode_plus(temp, max_length = 512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", )        
        token_count = torch.count_nonzero(temp_source['input_ids'], axis = 1)   
#         print("TOKEN COUNT: ", token_count)
        idx = torch.argmin(token_count % self.max_source_len)
#         print("IDX: ",idx)
        source_text = str(temp[idx])
        source_text = " ".join(source_text.split())
#         print("ext sum: ", source_text)
        source = tokenizer.batch_encode_plus([source_text], max_length = 512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt" )
        source_ids, source_mask = source["input_ids"].squeeze(), source["attention_mask"].squeeze()
#         print("========== BEFORE ==========")
#         print(source_ids)
        source_ids, source_ids_short, source_mask = self._get_head(source_ids, source_mask, self.max_source_len) 
#         print("========== AFTER ==========")
#         print(source_ids)
        return source_ids, source_ids_short, source_mask