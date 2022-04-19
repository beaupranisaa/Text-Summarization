import os
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'
from config import *
from score import  RougeScore
import time 

start_time = time.time()

paths = [
         f"""model/{model_params["MODEL"]}_{data}_head+tail25_50epochs_ratio2080/""",
         f"""model/{model_params["MODEL"]}_{data}_head+tail35_50epochs_ratio2080/""",
         f"""model/{model_params["MODEL"]}_{data}_head+tail45_50epochs_ratio2080/""",         
]

for path in paths:
#     path = f"""model/{model_params["MODEL"]}_{data}_tail-only35_50epochs/"""
    print("PATH: ", path)
    rougescore = RougeScore(model = model_params["MODEL"], dataset = data, EPOCH = model_params['TRAIN_EPOCHS'], path = path)
    # rougescore.path_gen = f'{model}_{dataset}_nomask/outputs_{model}_{dataset}/results_gen'
    # rougescore.path_eval = f'{model}_{dataset}_nomask/outputs_{model}_{dataset}/results_eval'
    rougescore.calculate()

    print("--- %s seconds ---" % (time.time() - start_time)) #ca 2163 sec = 36 mins