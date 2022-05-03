import os
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'
from datasets import load_dataset
dataset = load_dataset("xsum")
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = dataset['train']['document']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
a = list(vectorizer.get_feature_names_out())
print(a)
a.reverse()
print(len(a))
print(a)