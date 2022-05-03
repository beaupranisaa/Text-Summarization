import os
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'
from config import *
from score import  BertScore
import time 

start_time = time.time()

paths = [
         f"""model/{model_params["MODEL"]}_{data}_stopwords_all_50epochs/""",       
         f"""model/{model_params["MODEL"]}_{data}_stopwords_neg_50epochs/""",
]



for path in paths:
#     path = f"""model/{model_params["MODEL"]}_{data}_tail-only25_50epochs/"""
    print("PATH: ", path)
    bertscore = BertScore(model = model_params["MODEL"], dataset = data, EPOCH = model_params['TRAIN_EPOCHS'], path = path)
    #     bertscore.path_gen = f'{models[i]}_{datasets[i]}_nomask/outputs_{models[i]}_{datasets[i]}/results_gen'
    #     bertscore.path_eval = f'{models[i]}_{datasets[i]}_nomask/outputs_{models[i]}_{datasets[i]}/results_eval'
    bertscore.calculate()
    print("--- %s seconds ---" % (time.time() - start_time)) # ca 2.5 mins