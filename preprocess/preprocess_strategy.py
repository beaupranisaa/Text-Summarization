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

from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import T5Tokenizer

class SentenceLevelStrategy:
    def __init__(self, source_text, source_ids, source_len, max_source_len, mode = None):
        self.source_text = source_text
        self.source_ids = source_ids
        self.source_len = len(source_len["input_ids"].squeeze())
        self.end_eos = int(torch.where(self.source_ids == 1)[0])
        self.max_source_len = max_source_len
        self.mode = mode
    
    def shorten(self, method):
        if "luhn" in method:
            source_text_short, source_text_short_len = self._get_luhn(self.source_text, self.source_ids, self.max_source_len, self.mode)
            
        elif "textrank" in method:
            source_text_short, source_text_short_len = self._get_textrank(self.source_text, self.source_ids, self.max_source_len, self.mode)
            
        elif "lsa" in method:
            source_text_short, source_text_short_len = self._get_lsa(self.source_text, self.source_ids, self.max_source_len, self.mode)
        
        elif "stopwords" in method:
            source_text_short, n_stopwords = self._get_stopwords_removed(self.source_text, self.mode)
            return source_text_short, n_stopwords
        
        elif "tfidf" in method:
            source_text_short, n_stopwords = self._get_tfidf_removed(self.source_text, self.max_source_len)
            return source_text_short, n_stopwords
        
        else:
            raise ValueError("Undefined strategy ...") 
        
        return source_text_short, source_text_short_len
    
    def _get_luhn(self, source_text, source_ids, max_source_len, mode):
        if mode == "combo": 
            sentence_count = source_text.count(".") + 1
        else: sentence_count = source_text.count("\n") + 1 # count number of sentences
        token_count = torch.count_nonzero(source_ids)
        n_token_per_sentence = token_count/sentence_count
        n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
        parser = PlaintextParser.from_string(source_text[11:],Tokenizer("english")) #remove summarize: 
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
        parser = PlaintextParser.from_string(source_text[11:],Tokenizer("english")) #remove summarize: 
        summarizer = TextRankSummarizer()
        summary, summary_len = self._get_summary(summarizer, parser, n_sum_sentence)
        return  summary, summary_len

    def _get_lsa(self, source_text, source_ids, max_source_len, mode):
        sentence_count = source_text.count("\n") + 1 # count number of sentences
        token_count = torch.count_nonzero(source_ids)
        n_token_per_sentence = token_count/sentence_count
        n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
        parser = PlaintextParser.from_string(source_text[11:],Tokenizer("english")) #remove summarize: 
        summarizer = LsaSummarizer()
        summary, summary_len = self._get_summary(summarizer, parser, n_sum_sentence)
        return  summary, summary_len
    
    def _get_summary(self, summarizer, parser, n_sum_sentence):    
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        sum_candidates = []
#         print("HI: ",n_sum_sentence)
        for i in range(n_sum_sentence-1, n_sum_sentence +2):
            summary = summarizer(parser.document,i)
            full_summary = ' '.join([sentence._text for sentence in summary])
            sum_candidates.append(full_summary)
        
        source = tokenizer.batch_encode_plus(sum_candidates, max_length = 512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", )        
        token_count = torch.count_nonzero(source['input_ids'], axis = 1)   
        idx = torch.argmin(token_count % self.max_source_len)
        return sum_candidates[idx], token_count[idx]

    def _get_stopwords_removed(self, source_text):
        stop_words = list(set(stopwords.words('english')))
        # to discard negations --> please uncomment
#         negations = ["wouldn't", "hasn", 'not', "shan't", "didn't", "couldn't", 'aren',  'didn',"mustn't", "couldn", "don", "isn't",
#                      'mightn',   'mustn', "shouldn't", 'wasn',"shan", 'weren', "nor", 'needn', "hadn't", "wasn't", "shouldn", "won't", 
#                      "doesn't","won", "isn", "doesn", "hasn't", "weren't","needn't",  "haven't", "no", "mightn't", "wouldn", "aren't", "don't"]
#         for neg in negations:
#             stop_words.remove(neg)
        word_tokens = word_tokenize(source_text[11:])
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        n_stopwords = len(word_tokens) - len(filtered_sentence)
        filtered_sentence = " ".join(filtered_sentence)
        return filtered_sentence, n_stopwords
    
    def _get_tfidf_removed(self, source_text, max_source_len):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        corpus = [source_text[11:]]
        word_tokens = word_tokenize(source_text[11:])
        word_tokens = [ w.lower() for w in word_tokens]
        print("LEN WORD TOKENS: ", len(word_tokens))
        n = len(word_tokens) - max_source_len
        print("LEN TOKEN DIFF: ", n)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        tfidf_words = list(vectorizer.get_feature_names_out())
        print(tfidf_words)
        print("LEN TFIDF: ", len(tfidf_words))
        for tfidf in tfidf_words[:math.floor(3.5*n)]:
            if tfidf in word_tokens:
                print(tfidf)
                word_tokens.remove(tfidf)
                print(len(word_tokens))
        print(word_tokens)
        word_tokens = " ".join(word_tokens)
        source = tokenizer.batch_encode_plus([word_tokens], max_length = 512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", )  
        token_count = torch.count_nonzero(source['input_ids'])   
        print(token_count)
#         print(len(filtered_sentence))
#         n_stopwords = len(word_tokens) - len(filtered_sentence)
#         filtered_sentence = " ".join(filtered_sentence)
        return filtered_sentence, n_stopwords