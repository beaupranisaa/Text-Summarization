data = 'xsum'
params = {
    "MODEL": "t5-small",
    "BATCH_SIZE": 128,  # training batch size
    "SHORTENING QUANTITY": "all",
    "MAX_SOURCE_TEXT_LENGTH": '-',  # max length of source text 25% 373/ 35% 323/ 45% 273
    "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
    "METHOD": "stopwords", # luhn, lsa, textrank
    "RESTRICTION" : True,
}
