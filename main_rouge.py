import os
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'
from config import *
from score import  RougeScore

path = f"""model/{model_params["MODEL"]}_{data}_head-only_25/"""
rougescore = RougeScore(model = model_params["MODEL"], dataset = data, EPOCH = model_params['TRAIN_EPOCHS'], path = path)
# rougescore.path_gen = f'{model}_{dataset}_nomask/outputs_{model}_{dataset}/results_gen'
# rougescore.path_eval = f'{model}_{dataset}_nomask/outputs_{model}_{dataset}/results_eval'
rougescore.calculate()