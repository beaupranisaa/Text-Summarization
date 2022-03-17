import os
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'
from config import *
from score import  BertScore

bertscore = BertScore(model = model_params["MODEL"], dataset = data, EPOCH = model_params['TRAIN_EPOCHS'], path = path)
#     bertscore.path_gen = f'{models[i]}_{datasets[i]}_nomask/outputs_{models[i]}_{datasets[i]}/results_gen'
#     bertscore.path_eval = f'{models[i]}_{datasets[i]}_nomask/outputs_{models[i]}_{datasets[i]}/results_eval'
bertscore.calculate()