import os
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'
from config import *
from score import  RougeScore
import time 

start_time = time.time()




paths = [
#          f"""model/{model_params["MODEL"]}_{data}_stopwords_neg_head+tail_ratio0.5_25_50epochs/""",   
#          f"""model/{model_params["MODEL"]}_{data}_stopwords_neg_head+tail_ratio0.5_35_50epochs/""",
#          f"""model/{model_params["MODEL"]}_{data}_stopwords_neg_head+tail_ratio0.5_45_50epochs/""",
         f"""model/{model_params["MODEL"]}_{data}_combo_stopwords_neg_luhn_25_50epochs/""",
         f"""model/{model_params["MODEL"]}_{data}_combo_stopwords_neg_luhn_35_50epochs/""",
         f"""model/{model_params["MODEL"]}_{data}_combo_stopwords_neg_luhn_45_50epochs/""", 
         f"""model/{model_params["MODEL"]}_{data}_combo_stopwords_all_luhn_25_50epochs/""",
         f"""model/{model_params["MODEL"]}_{data}_combo_stopwords_all_luhn_35_50epochs/""",
         f"""model/{model_params["MODEL"]}_{data}_combo_stopwords_all_luhn_45_50epochs/""",  
         f"""model/{model_params["MODEL"]}_{data}_combo_stopwords_all_textrank_25_50epochs/""",
         f"""model/{model_params["MODEL"]}_{data}_combo_stopwords_all_textrank_35_50epochs/""",
         f"""model/{model_params["MODEL"]}_{data}_combo_stopwords_all_textrank_45_50epochs/""",  
]
for path in paths:
#     path = f"""model/{model_params["MODEL"]}_{data}_tail-only35_50epochs/"""
    print("PATH: ", path)
    rougescore = RougeScore(model = model_params["MODEL"], dataset = data, EPOCH = model_params['TRAIN_EPOCHS'], path = path)
    # rougescore.path_gen = f'{model}_{dataset}_nomask/outputs_{model}_{dataset}/results_gen'
    # rougescore.path_eval = f'{model}_{dataset}_nomask/outputs_{model}_{dataset}/results_eval'
    rougescore.calculate()

    print("--- %s seconds ---" % (time.time() - start_time)) #ca 2163 sec = 36 mins