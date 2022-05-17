data = 'xsum'

model_params = {
    "MODEL": "t5-small",  # model_type: t5-base/t5-large/longformer-base-4096/bart-base
    "TRAIN_BATCH_SIZE": 16,  # training batch size
    "VALID_BATCH_SIZE": 16,  # validation batch size
    "TRAIN_EPOCHS": 90,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 2e-05,  # learning rate default betas=(0.9, 0.999), eps=1e-08
    "SCHEDULER": "linear",
    "SHORTENING QUANTITY": 'all',
    "MAX_SOURCE_TEXT_LENGTH": '-',  # max length of source text 25% 373/ 35% 323/ 45% 273/ 'all' '-'/ 
    "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
    "METHOD": "full-text", # full-text, head-only, tail-only, head+tail_ratio0.2, head+tail_ratio0.5 ,combo_stopwords_all_luhn
    "PATH": None, #reprocess/preprocessed_text/textrank_1024/quantity_512/
    "RESTRICTION": True,
    "RESUME_FROM_CHECKPOINTS": True
}

path = f"""./model/{model_params["MODEL"]}_{data}_{model_params["METHOD"]}_{model_params["SHORTENING QUANTITY"]}_50epochs_testnewtrain/"""

mask = False
to_mask_list = None

# path must be None if method == full-text, head-only, tail-only, head+tail_ratio0.2, head+tail_ratio0.5