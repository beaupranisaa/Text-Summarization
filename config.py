data = 'xsum'

model_params = {
    "MODEL": "t5-small",  # model_type: t5-base/t5-large/longformer-base-4096/bart-base
    "TRAIN_BATCH_SIZE": 16,  # training batch size
    "VALID_BATCH_SIZE": 16,  # validation batch size
    "TRAIN_EPOCHS": 50,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 2e-05,  # learning rate default betas=(0.9, 0.999), eps=1e-08
    "SCHEDULER": "linear",
    "SHORTENING QUANTITY": 0,
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text 25% 373/ 35% 323/ 45% 273
    "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
    "METHOD": "full-text", # full-text, head-only, tail-only, head+tail_ratio0.2, head+tail_ratio0.5
}

path = f"""./model/{model_params["MODEL"]}_{data}_{model_params["METHOD"]}_{model_params["SHORTENING QUANTITY"]}_{model_params["TRAIN_EPOCHS"]}epochs_test/"""

len_restriction = True
mask = False
to_mask_list = None
resume_from_checkpoint = True