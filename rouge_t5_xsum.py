import os
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

from eval_new import  RougeScore
model = "t5-small"
dataset = "xsum"
rougescore = RougeScore(model = "t5-small", dataset = "xsum", EPOCH = 50)
rougescore.path_gen = f'{model}_{dataset}_nomask/outputs_{model}_{dataset}/results_gen'
rougescore.path_eval = f'{model}_{dataset}_nomask/outputs_{model}_{dataset}/results_eval'
rougescore.calculate()