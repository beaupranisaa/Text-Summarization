data = 'xsum'

model_params = {
    "MODEL": "t5-small",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 16,  # training batch size
    "VALID_BATCH_SIZE": 16,  # validation batch size
    "TRAIN_EPOCHS": 30,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 2e-05,  # learning rate default betas=(0.9, 0.999), eps=1e-08
    "SCHEDULER": "linear",
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}

# path = f"""./model/{model_params["MODEL"]}_{data}_nolenrestriction/"""
path = f"""./model/{model_params["MODEL"]}_{data}_nolenrestriction_30epoch/"""

len_restriction = False
mask = False
to_mask_list = None