data = 'xsum'

model_params = {
    "MODEL": "t5-small",  # model_type: t5-base/t5-large/longformer-base-4096/bart-base
    "TRAIN_BATCH_SIZE": 16,  # training batch size
    "VALID_BATCH_SIZE": 16,  # validation batch size
    "TRAIN_EPOCHS": 50,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 2e-05,  # learning rate default betas=(0.9, 0.999), eps=1e-08
    "SCHEDULER": "linear",
    "SHORTENING QUANTITY": 25,
    "MAX_SOURCE_TEXT_LENGTH": 373,  # max length of source text 25% 373/ 35% 323/ 45% 273
    "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
    "METHOD": "head-only", # full-text, head-only, tail-only, head+tail_ratio0.2, head+tail_ratio0.5
}


# assert model_params["SHORTENING QUANTITY"] in [0, 25, 35, 45]
# assert model_params["MAX_SOURCE_TEXT_LENGTH"] in [512, 373, 323, 273]
# assert model_params["MAX_SOURCE_TEXT_LENGTH"] in ["full-text", "head-only", "tail-only",]+["head+tail_ratio{:.1f}".format(i) for i in np.arange(0.0, 1.0, 0.1)]

# if model_params["METHOD"] == "full-text":
#     assert model_params["SHORTENING QUANTITY"] == 0
#     assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 512

# if model_params["SHORTENING QUANTITY"] == 25:
#     assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 373
# elif model_params["SHORTENING QUANTITY"] == 35:
#     assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 323
# elif model_params["SHORTENING QUANTITY"] == 45:
#     assert model_params["MAX_SOURCE_TEXT_LENGTH"] == 273
# else:
#     pass
    
# path = f"""./model/{model_params["MODEL"]}_{data}_nolenrestriction/"""
path = f"""./model/{model_params["MODEL"]}_{data}_{model_params["METHOD"]}_{model_params["SHORTENING QUANTITY"]}_{model_params["TRAIN_EPOCHS"]}epochs/"""

len_restriction = True
mask = False
to_mask_list = None