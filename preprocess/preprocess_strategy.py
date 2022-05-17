import math
import re
import torch

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

from summarizer import Summarizer #bertbase

from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import T5Tokenizer

class SentenceLevelStrategy:
    def __init__(self, source_text, source_ids, source_len, max_source_len, mode = None):
        self.source_text = source_text
        self.source_ids = source_ids
        self.source_len = source_len
        self.end_eos = int(torch.where(self.source_ids == 1)[0])
        self.max_source_len = max_source_len
        self.mode = mode
    
    def shorten(self, method):
        if "full-text" in method:
            source_text_short, source_text_short_len = self._get_fulltext(self.source_ids, self.source_len)
        
        elif "head-only" in method:
            source_text_short, source_text_short_len = self._get_head(self.source_ids, self.max_source_len)
        
        elif "tail-only" in method:
            source_text_short, source_text_short_len = self._get_tail(self.source_ids, self.end_eos, self.max_source_len)
        
        elif "head+tail" in method:
            head_ratio = float(re.findall(r"\d*\.\d", method)[0])
            print(head_ratio)
            source_text_short, source_text_short_len = self._get_headtail(self.source_ids, self.end_eos, self.max_source_len, head_ratio)

        elif "luhn" in method:
            source_text_short, source_text_short_len = self._get_luhn(self.source_text, self.source_ids, self.max_source_len, self.mode)
            
        elif "textrank" in method:
            source_text_short, source_text_short_len = self._get_textrank(self.source_text, self.source_ids, self.max_source_len, self.mode)
            
        elif "lsa" in method:
            source_text_short, source_text_short_len = self._get_lsa(self.source_text, self.source_ids, self.max_source_len, self.mode)
        
        elif "stopwords" in method:
            source_text_short, source_text_short_len = self._get_stopwords_removed(self.source_text, self.mode)

        elif "bertbased":
            source_text_short, source_text_short_len = self._get_bertbased(self.source_text, self.source_ids, self.max_source_len, self.mode)
        else:
            raise ValueError("Undefined strategy ...") 
        
        return source_text_short, source_text_short_len
    
#     def _get_fulltext(self, source_ids, source_len):
#         source_ids_short = source_ids[0:source_len] # the first two tokens are "summarize" and ":"
#         tokenizer = T5Tokenizer.from_pretrained("t5-small")
#         shortened_source_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in source_ids_short]
#         return shortened_source_text, source_len
    
    def _get_head(self, source_ids, max_source_len):
        source_ids_short = source_ids[0:max_source_len]
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        shortened_source_text = tokenizer.decode(source_ids_short, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return shortened_source_text, torch.count_nonzero(source_ids_short.to(dtype=torch.long))
    
    def _get_tail(self, source_ids, end_eos, max_source_len):
        source_ids_short = source_ids[end_eos-max_source_len+1:end_eos+1] 
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        shortened_source_text = tokenizer.decode(source_ids_short, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return shortened_source_text, torch.count_nonzero(source_ids_short.to(dtype=torch.long))
    
    def _get_headtail(self, source_ids, end_eos, max_source_len, head_ratio):
        print("Hi")
        head_ids = source_ids[0:math.floor(head_ratio*max_source_len)+1]
        tail_ids = source_ids[end_eos - math.floor((1-head_ratio)*max_source_len)+1:end_eos+1]
        source_ids_short = torch.cat((head_ids, tail_ids),0)
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        shortened_source_text = tokenizer.decode(source_ids_short, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return shortened_source_text, torch.count_nonzero(source_ids_short.to(dtype=torch.long))
    
    def _get_luhn(self, source_text, source_ids, max_source_len, mode):
        if mode == "combo": 
            sentence_count = source_text.count(".") + 1
        else: sentence_count = source_text.count("\n") + 1 # count number of sentences
        token_count = torch.count_nonzero(source_ids)
        n_token_per_sentence = token_count/sentence_count
        n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
        parser = PlaintextParser.from_string(source_text,Tokenizer("english"))
        summarizer = LuhnSummarizer()
        summary, summary_len = self._get_summary(summarizer, parser, n_sum_sentence)
        return  summary, summary_len

    def _get_textrank(self, source_text, source_ids, max_source_len, mode):
        if mode == "combo": 
            sentence_count = source_text.count(".") + 1
        else: sentence_count = source_text.count("\n") + 1 # count number of sentences
        token_count = torch.count_nonzero(source_ids)
        n_token_per_sentence = token_count/sentence_count
        n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
        parser = PlaintextParser.from_string(source_text,Tokenizer("english")) #remove summarize: 
        summarizer = TextRankSummarizer()
        summary, summary_len = self._get_summary(summarizer, parser, n_sum_sentence)
        return  summary, summary_len

    def _get_lsa(self, source_text, source_ids, max_source_len, mode):
        sentence_count = source_text.count("\n") + 1 # count number of sentences
        token_count = torch.count_nonzero(source_ids)
        n_token_per_sentence = token_count/sentence_count
        n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
        parser = PlaintextParser.from_string(source_text,Tokenizer("english")) #remove summarize: 
        summarizer = LsaSummarizer()
        summary, summary_len = self._get_summary(summarizer, parser, n_sum_sentence)
        return  summary, summary_len
    
    def _get_summary(self, summarizer, parser, n_sum_sentence):    
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        sum_candidates = []
        for i in range(n_sum_sentence-1, n_sum_sentence +2):
            summary = summarizer(parser.document,i)
            full_summary = ' '.join([sentence._text for sentence in summary])
            sum_candidates.append(full_summary)
        source = tokenizer.batch_encode_plus(sum_candidates, max_length = 512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", )        
        token_count = torch.count_nonzero(source['input_ids'], axis = 1)   
        idx = torch.argmin(token_count % self.max_source_len)
        return sum_candidates[idx], token_count[idx]

    def _get_stopwords_removed(self, source_text, mode):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        stop_words = list(set(stopwords.words('english')))
#         print(stop_words)
        word_tokens = word_tokenize(source_text)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        n_stopwords = len(word_tokens) - len(filtered_sentence)
        filtered_sentence = " ".join(filtered_sentence)
#         print(filtered_sentence)
        edited = "% ".join([a.strip() for a in filtered_sentence.split("%")])
        edited = " [".join([a.strip() for a in edited.split("[ ")])
        edited = "] ".join([a.strip() for a in edited.split(" ]")])
        edited = " (".join([a.strip() for a in edited.split("( ")])
        edited = ") ".join([a.strip() for a in edited.split(" )")])
        edited = ". ".join([a.strip() for a in edited.split(".")])
        edited = ", ".join([a.strip() for a in edited.split(",")])
        edited = ": ".join([a.strip() for a in edited.split(":")])
        edited = ".".join([a.strip() for a in edited.split(" .")])
        edited = "? ".join([a.strip() for a in edited.split("?")])
        edited = edited.replace("``", "")
        edited = edited.replace("''", "")
        edited = edited.replace("  ", " ")
#         print(edited)
        source = tokenizer.batch_encode_plus([edited], max_length = 512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", ) 
        token_count = torch.count_nonzero(source['input_ids'], axis = 1)   
        return edited, token_count[0]
    
    def _get_bertbased(self,source_text, source_ids, max_source_len, mode):
        sentence_count = source_text.count("\n") + 1 # count number of sentences
        token_count = torch.count_nonzero(source_ids)
        n_token_per_sentence = token_count/sentence_count
        n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
        summarizer = Summarizer()
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        sum_candidates = []
#         print("HI: ",n_sum_sentence)
        for i in range(n_sum_sentence-1, n_sum_sentence +2):
            summary = summarizer(source_text, num_sentences = i)
            full_summary = ''.join(summary)
            sum_candidates.append(full_summary)
        source = tokenizer.batch_encode_plus(sum_candidates, max_length = 512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", )        
        token_count = torch.count_nonzero(source['input_ids'], axis = 1)   
        idx = torch.argmin(token_count % self.max_source_len)
        return sum_candidates[idx], token_count[idx]
 

'''
# negations = ["wouldn't", "hasn", 'not', "shan't", "didn't", "couldn't", 'aren',  'didn',"mustn't", "couldn", "don", "isn't",'mightn',   'mustn', "shouldn't", 'wasn',"shan", 'weren', "nor", 'needn', "hadn't", "wasn't", "shouldn", "won't", 
#                      "doesn't","won", "isn", "doesn", "hasn't", "weren't","needn't",  "haven't", "no", "mightn't", "wouldn", "aren't", "don't"]
#         for neg in negations:
#             stop_words.remove(neg)
'''