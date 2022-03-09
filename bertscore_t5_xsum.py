import os
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

from eval_new import  BertScore

models = ["t5-small"]
datasets = ["xsum"]

for i in range(len(models)):
    print("MODEL: ", models[i], "DATASET: ", datasets[i])
    bertscore = BertScore(model = models[i], dataset = datasets[i], EPOCH = 50)
    bertscore.path_gen = f'{models[i]}_{datasets[i]}_nomask/outputs_{models[i]}_{datasets[i]}/results_gen'
    bertscore.path_eval = f'{models[i]}_{datasets[i]}_nomask/outputs_{models[i]}_{datasets[i]}/results_eval'
    bertscore.calculate()