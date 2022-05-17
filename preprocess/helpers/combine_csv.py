import sys         
sys.path.append('/home/pranisaa/working_dir/Text-Summarization/preprocess')
import os
from preprocess_config import *
import pandas as pd

# params = {
#     "MODEL": "t5-small",
#     "BATCH_SIZE": 128,  # training batch size
#     "SHORTENING QUANTITY": "all",
#     "MAX_SOURCE_TEXT_LENGTH": '-',  # max length of source text 25% 373/ 35% 323/ 45% 273
#     "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
#     "SEED": 42,  # set seed for reproducibility
#     "METHOD": "stopwords", # luhn, lsa, textrank
#     "RESTRICTION" : True,
# }

# def combine(params, mode = "train_set"):
#     path = f"""../preprocessed_text/{params["METHOD"]}/quantity_{params["SHORTENING QUANTITY"]}/{mode}/"""
#     content = os.listdir(path)
#     preprocessed_text_step = [path for path in content if ".csv" in path]
#     if params["METHOD"] == "stopwords":
#         final = {"Sample ids" : [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Removed words": []}
#     else:
#         final = {"Sample ids" : [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] }
#     for i in range(len(preprocessed_text_step)):
#         df = pd.read_csv(os.path.join(path,f"""{params["METHOD"]}_quantity{params["SHORTENING QUANTITY"]}_step{i}.csv""" ))
#         final["Sample ids"].extend(df["Sample ids"])
#         final["Document"].extend(df["Document"])
#         final["Shortened Document"].extend(df["Shortened Document"])
#         final["Summary"].extend(df["Summary"])
#         final["Document length"].extend(df["Document length"])
#         if params["METHOD"] == "stopwords":
#             final["Removed words"].extend(df["Removed words"])
#     final_df = pd.DataFrame(final)
#     final_df.to_csv(os.path.join(f"""../preprocessed_text/{params["METHOD"]}/quantity_{params["SHORTENING QUANTITY"]}""", f"""{mode}.csv"""))
#     print("SAVE TO CSV FINISHED")


config = dict(
    model = "t5-small",
    data = "xsum",    
    batch_size = 128,
    orig_source_length = 512,
    max_source_length = 512, # 512, 373, 323, 273
    max_target_length = 36,
    method = "stopwords", #head+tail0.5
    seed = 42,
)
     

def combine(config, mode = "train_set"):
    path = f"""../preprocessed_text/{config['orig_source_length']}_{config['method']}/quantity_{config['max_source_length']}"""
    print("PATH: ", path)
    content = os.listdir(os.path.join(path,mode)) 
    preprocessed_text_step = [path for path in content if ".csv" in path]
    final = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Shortened Document length": []} 
    
    for i in range(len(preprocessed_text_step)):
        df = pd.read_csv(os.path.join(path, f"""{mode}/step{i}.csv"""""))
        final["Sample ids"].extend(df["Sample ids"])
        final["Document"].extend(df["Document"])
        final["Shortened Document"].extend(df["Shortened Document"])
        final["Summary"].extend(df["Summary"])
        final["Document length"].extend(df["Document length"])
        final["Shortened Document length"].extend(df["Shortened Document length"])
    if mode == "train_set":
        assert len(final["Sample ids"]) == 4795
    elif mode == "val_set":
        assert len(final["Sample ids"]) == 273
    else:
        assert len(final["Sample ids"]) == 266
#     print(f"""{mode} LEN: {len(final["Sample ids"])}""")
    final_df = pd.DataFrame(final)
    final_df.to_csv(os.path.join(path, f"""{mode}.csv"""))
    print("SAVE TO CSV FINISHED")

if __name__ == "__main__":
#     print(f""" METHOD: {config['method']} SOURCE LENGTH: {config['max_source_length']}""")
    combine(config, mode = "train_set")
    combine(config, mode = "val_set")
    combine(config, mode = "test_set")
    