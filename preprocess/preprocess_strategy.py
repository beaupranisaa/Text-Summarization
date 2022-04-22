import math
import re
import torch

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer

from transformers import T5Tokenizer

class SentenceLevelStrategy:
    def __init__(self, source_text, source_ids, source_len, max_source_len):
        self.source_text = source_text
        self.source_ids = source_ids
        self.source_len = len(source_len["input_ids"].squeeze())
        self.end_eos = int(torch.where(self.source_ids == 1)[0])
        self.max_source_len = max_source_len
    
    def shorten(self, method):
        if "luhn" in method:
            source_text_short = self._get_luhn(self.source_text, self.source_ids, self.max_source_len)
            
        elif "textrank" in method:
            source_text_short = self._get_textrank(self.source_text, self.source_ids, self.max_source_len)
            
        elif "lsa" in method:
            source_text_short = self._get_lsa(self.source_text, self.source_ids, self.max_source_len)
        else:
            pass 
        
        return source_text_short
    
    def _get_luhn(self, source_text, source_ids, max_source_len):
        sentence_count = source_text.count("\n") + 1 # count number of sentences
        token_count = torch.count_nonzero(source_ids)
        n_token_per_sentence = token_count/sentence_count
        n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
        parser = PlaintextParser.from_string(source_text[11:],Tokenizer("english")) #remove summarize: 
        summarizer = LuhnSummarizer()
        summary = self._get_summary(summarizer, parser, n_sum_sentence)
        return  summary

    def _get_textrank(self, source_text, source_ids, max_source_len):
        sentence_count = source_text.count("\n") + 1 # count number of sentences
        token_count = torch.count_nonzero(source_ids)
        n_token_per_sentence = token_count/sentence_count
        n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
        parser = PlaintextParser.from_string(source_text[11:],Tokenizer("english")) #remove summarize: 
        summarizer = TextRankSummarizer()
        summary = self._get_summary(summarizer, parser, n_sum_sentence)
        return  summary

    def _get_lsa(self, source_text, source_ids, max_source_len):
        sentence_count = source_text.count("\n") + 1 # count number of sentences
        token_count = torch.count_nonzero(source_ids)
        n_token_per_sentence = token_count/sentence_count
        n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
        parser = PlaintextParser.from_string(source_text[11:],Tokenizer("english")) #remove summarize: 
        summarizer = LsaSummarizer()
        summary = self._get_summary(summarizer, parser, n_sum_sentence)
        return  summary
    
    def _get_summary(self, summarizer, parser, n_sum_sentence):    
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        sum_candidates = []
#         print("HI: ",n_sum_sentence)
        for i in range(n_sum_sentence-1, n_sum_sentence +2):
            summary = summarizer(parser.document,i)
            full_summary = ' '.join([sentence._text for sentence in summary])
#             full_summary = "summarize: " + full_summary
            sum_candidates.append(full_summary)
        
        source = tokenizer.batch_encode_plus(sum_candidates, max_length = 512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", )        
        token_count = torch.count_nonzero(source['input_ids'], axis = 1)   
#         print("TOKEN COUNT: ", token_count)
        idx = torch.argmin(token_count % self.max_source_len)
        return sum_candidates[idx]