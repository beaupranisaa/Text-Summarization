data = 'xsum'
params = {
    "MODEL": "t5-small",
    "BATCH_SIZE": 128,  # training batch size
    "SHORTENING QUANTITY": 512,
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text 25% 373/ 35% 323/ 45% 273
    "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
    "METHOD": "textrank", # luhn, lsa, textrank
    "RESTRICTION" : True,
}

params_combo = {
    "MODEL": "t5-small",
    "BATCH_SIZE": 128,  # training batch size
    "PATH": "preprocessed_text/stopwords_1024/",
    "METHOD1": "stopwords_1024",
    "SHORTENING QUANTITY1": "all",
    "METHOD2": "textrank",
    "SHORTENING QUANTITY2": 512,    
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text 25% 373/ 35% 323/ 45% 273
    "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
    "RESTRICTION" : True,
}


