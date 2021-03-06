import sys         
sys.path.append('/home/pranisaa/working_dir/Text-Summarization/preprocess')
import os
from preprocess_config import *
import pandas as pd

params_combo = {
    "MODEL": "t5-small",
    "BATCH_SIZE": 128,  # training batch size
    "PATH": "../preprocessed_text/combo_stopwords_1024_all_luhn/",
    "METHOD1": "stopwords_1024",
    "SHORTENING QUANTITY1": "all",
    "METHOD2": "luhn",
    "SHORTENING QUANTITY2": 512,    
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text 25% 373/ 35% 323/ 45% 273
    "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
    "RESTRICTION" : True,
}



def combine(params, mode = "train_set"):
    path = os.path.join(params["PATH"],f"""quantity_{params["SHORTENING QUANTITY2"]}/{mode}""")
    content = os.listdir(path)
    preprocessed_text_step = [path for path in content if ".csv" in path]
    if params["METHOD2"] == "stopwords":
        final = {"Sample ids" : [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Removed words": []}
    else:
        final = {"Sample ids" : [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Shortened Document len": []}
    for i in range(len(preprocessed_text_step)):
        df = pd.read_csv(os.path.join(path,f"""combo_{params["METHOD1"]}_{params["SHORTENING QUANTITY1"]}_{params["METHOD2"]}_{params["SHORTENING QUANTITY2"]}step{i}.csv""" ))
        final["Sample ids"].extend(df["Sample ids"])
        final["Document"].extend(df["Document"])
        final["Shortened Document"].extend(df["Shortened Document"])
        final["Summary"].extend(df["Summary"])
        final["Document length"].extend(df["Document length"])
        final["Shortened Document len"].extend(df["Shortened Document len"])
        if params["METHOD2"] == "stopwords":
            final["Removed words"].extend(df["Removed words"])
    final_df = pd.DataFrame(final)
    final_df.to_csv(os.path.join(f"""../preprocessed_text/combo_{params["METHOD1"]}_{params["SHORTENING QUANTITY1"]}_{params["METHOD2"]}/quantity_{params["SHORTENING QUANTITY2"]}""", f"""{mode}.csv"""))
    print("SAVE TO CSV FINISHED")
    
if __name__ == "__main__":  
    print()
    combine(params_combo, mode = "train_set")
    combine(params_combo, mode = "test_set")
