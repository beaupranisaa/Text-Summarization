import sys         
sys.path.append('/home/pranisaa/working_dir/Text-sum-test/preprocess')
import os
from preprocess_config import *
import pandas as pd

def combine(params, mode = "train_set"):
    path = f"""../preprocessed_text/{params["METHOD"]}/quantity_{params["SHORTENING QUANTITY"]}/{mode}/"""
    content = os.listdir(path)
    preprocessed_text_step = [path for path in content if ".csv" in path]
    if params["METHOD"] == "stopwords":
        final = {"Sample ids" : [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Removed words": []}
    else:
        final = {"Sample ids" : [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] }
    for i in range(len(preprocessed_text_step)):
        df = pd.read_csv(os.path.join(path,f"""{params["METHOD"]}_quantity{params["SHORTENING QUANTITY"]}_step{i}.csv""" ))
        final["Sample ids"].extend(df["Sample ids"])
        final["Document"].extend(df["Document"])
        final["Shortened Document"].extend(df["Shortened Document"])
        final["Summary"].extend(df["Summary"])
        final["Document length"].extend(df["Document length"])
        if params["METHOD"] == "stopwords":
            final["Removed words"].extend(df["Removed words"])
    final_df = pd.DataFrame(final)
    final_df.to_csv(os.path.join(f"""../preprocessed_text/{params["METHOD"]}/quantity_{params["SHORTENING QUANTITY"]}""", f"""{mode}.csv"""))
    print("SAVE TO CSV FINISHED")
    
if __name__ == "__main__":  
    
    combine(params, mode = "train_set")
    combine(params, mode = "test_set")
    