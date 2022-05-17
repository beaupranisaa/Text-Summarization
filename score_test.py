from datasets import load_metric
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import EvalDataset
import os
import matplotlib.pyplot as plt

class Scores:
    def __init__(self, model, dataset, path):
        self.params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
        }
        
        self.model = model
        self.dataset = dataset

        self.path = os.path.join(path, f"result_eval")
#         print(path)
    
    def calculate(self):
        pass
    
    def getscore(self):
        pass
    
    def plot(self):
        pass

class BertScore(Scores):
    def __init__(self, model, dataset, EPOCH, path):           
        Scores.__init__(self,model, dataset, path)
        
        self.EPOCH = EPOCH
        self.metric = load_metric("bertscore")
        
        self.bertscores = ['precision', 'recall', 'f1']
#         self.bertscore_calculate = {k: [] for k in bertscores}
        self.bertscore_get = {k: [] for k in self.bertscores}
        
    def calculate(self):
        for epoch in range(self.EPOCH):
            print("At: EPOCH ", epoch)
            self.bertscore_calculate = {k: [] for k in self.bertscores}
            data_set = EvalDataset(os.path.join(self.path, f"predictions_{self.model}_epoch{epoch}.csv"))
            loader = DataLoader(data_set, ** self.params)
#             print("DATA LOADED")
            for i,batch in enumerate(loader):
                ref, hyp = batch
                self.metric.add_batch(predictions=hyp, references=ref)
#                 if i % 1000 == 0:
#                     print("STEP: ", i, "/", len(loader))
            score = self.metric.compute(rescale_with_baseline = True, lang = 'en') #rescale_with_baseline=True
            self.bertscore_calculate['precision'] = score['precision']
            self.bertscore_calculate['recall'] = score['recall']
            self.bertscore_calculate['f1'] = score['f1']
            
            df = pd.read_csv(os.path.join(self.path, f"predictions_{self.model}_epoch{epoch}.csv"))
            
            df["BERTScore(precision)"] =  self.bertscore_calculate['precision']
            df["BERTScore(recall)"] = self.bertscore_calculate['recall']
            df["BERTScore(f1)"] = self.bertscore_calculate['f1']
            
            df.to_csv(os.path.join(self.path, f"predictions_{self.model}_epoch{epoch}.csv"), index=False)
            
            print("SAVED TO CSV: ", epoch)
            
    def getscore(self):
        for epoch in range(self.EPOCH):
            df = pd.read_csv(os.path.join(self.path, f"predictions_{self.model}_epoch{epoch}.csv"))
#             print(df)
            precision = np.mean(df["BERTScore(precision)"])
            recall = np.mean(df["BERTScore(recall)"])
            f1 = np.mean(df["BERTScore(f1)"])
            self.bertscore_get['precision'].append(precision)
            self.bertscore_get['recall'].append(recall)
            self.bertscore_get['f1'].append(f1)
        return self.bertscore_get
    
    def plot(self, toplot, label):
        plt.plot(toplot, label = label)
        plt.title(f"BERTScore({label})")
        plt.legend()
        plt.show()

class RougeScore(Scores):
    def __init__(self, model, dataset, EPOCH, path):           
        Scores.__init__(self,model, dataset, path)
    
        self.EPOCH = EPOCH
        self.metric = load_metric("rouge")
        
        self.rougescores = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        self.get_rouge = {k: [] for k in  self.rougescores}
        
    def calculate(self):
        for epoch in range(self.EPOCH): #self.EPOCH #range(13,50)
#         for epoch in self.EPOCH: # if list
            self.calculate_rouge = {k: [] for k in  self.rougescores}
            print("At: EPOCH ", epoch)
            data_set = EvalDataset(os.path.join(self.path, f"predictions_{self.model}_epoch{epoch}.csv"))
            loader = DataLoader(data_set, **self.params)
            print("DATA LOADED")
            for i,batch in enumerate(loader):
                ref, hyp = batch
                score = self.metric.compute(predictions=hyp, references=ref) 
                score = {key: value.mid.fmeasure * 100 for key, value in score.items()}
                self.calculate_rouge['rouge1'].append(score['rouge1']) 
                self.calculate_rouge['rouge2'].append(score['rouge2'])
                self.calculate_rouge['rougeL'].append(score['rougeL'])
                self.calculate_rouge['rougeLsum'].append(score['rougeLsum'])
                if i % 100 == 0:
                    print("STEP: ", i, "/", len(loader))
            df = pd.read_csv(os.path.join(self.path, f"predictions_{self.model}_epoch{epoch}.csv"))
            df["RougeScore1"] = self.calculate_rouge['rouge1']
            df["RougeScore2"] = self.calculate_rouge['rouge2']
            df["RougeScoreL"] = self.calculate_rouge['rougeL'] #ROUGE-L measures the longest common subsequence (LCS) between our model output and reference. 
            df["RougeScoreLsum"] = self.calculate_rouge['rougeLsum']
            
            df.to_csv(os.path.join(self.path, f"predictions_{self.model}_epoch{epoch}.csv"), index=False)
            print("SAVED TO CSV: ", epoch)
    
    def getscore(self):
        for epoch in range(self.EPOCH):
#         for epoch in self.EPOCH: # if list
            df = pd.read_csv(os.path.join(self.path, f"predictions_{self.model}_epoch{epoch}.csv"))
#             print(df)
            rouge1 = np.mean(df["RougeScore1"])
            rouge2 = np.mean(df["RougeScore2"])
            rougeL = np.mean(df["RougeScoreL"])
            rougeLsum = np.mean(df["RougeScoreLsum"])
            
            self.get_rouge['rouge1'].append(rouge1)
            self.get_rouge['rouge2'].append(rouge2)
            self.get_rouge['rougeL'].append(rougeL)
            self.get_rouge['rougeLsum'].append(rougeLsum)
        return self.get_rouge
    
    def getscore_ave(self):
        get_ave_rouge = {k: [] for k in  self.rougescores}
        for epoch in range(self.EPOCH):
#         for epoch in self.EPOCH: # if list
            df = pd.read_csv(os.path.join(self.path, f"rouge_{self.model}_epoch{epoch}.csv"))
            
            get_ave_rouge["rouge1"].append(df.iloc[0,1])
            get_ave_rouge["rouge2"].append(df.iloc[1,1])
            get_ave_rouge["rougeL"].append(df.iloc[2,1])
            get_ave_rouge["rougeLsum"].append(df.iloc[3,1])
        
        return get_ave_rouge   
    
    def plot(self, toplot, label):
        plt.plot(toplot, label = label)
        plt.title(label)
        plt.legend()
        plt.show()