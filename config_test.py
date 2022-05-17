# config = {
#     "MODEL": "t5-small",  # model_type: t5-base/t5-large/longformer-base-4096/bart-base
#     "DATA": 'xsum'
#     "BATCH_SIZE": 16,  # training batch size
#     "TRAIN_EPOCHS": 90,  # number of training epochs
#     "VAL_EPOCHS": 1,  # number of validation epochs
#     "LEARNING_RATE": 2e-05,  # learning rate default betas=(0.9, 0.999), eps=1e-08
#     "SCHEDULER": "linear",
#     "ORIG_SOURCE_TEXT_LENGTH": 512,
#     "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text 25% 373/ 35% 323/ 45% 273/ 'all' '-'/ 
#     "MAX_TARGET_TEXT_LENGTH": 36,  # max length of target text
#     "SEED": 42,  # set seed for reproducibility
#     "METHOD": "full-text_512", # full-text, head-only, tail-only, head+tail_ratio0.2, head+tail_ratio0.5 ,combo_stopwords_all_luhn
#     "PATH": None, #preprocess/preprocessed_text/textrank_1024/quantity_512/
#     "RESTRICTION": True,
#     "RESUME_FROM_CHECKPOINTS": True
# }

# path = f"""./model/{model_params["MODEL"]}_{data}_{model_params["METHOD"]}_{model_params["SHORTENING QUANTITY"]}_50epochs_testnewtrain/"""

# path must be None if method == full-text, head-only, tail-only, head+tail_ratio0.2, head+tail_ratio0.5


config = dict(
    model = "t5-small",
    data = "xsum",
    batch_size=16,
    train_epochs = 5,
    val_epochs = 1,
    learning_rate = 2e-04, # learning rate default betas=(0.9, 0.999), eps=1e-08
    scheduler = "linear",
    orig_source_length = 512,
    max_source_length = 512,
    max_target_length = 36,
    seed = 42,
    method = "stopwords",
    path = "preprocess/preprocessed_text/512_head-only/quantity_512/",
    restriction = True,
    resume_from_checkpoint = False,)

config["output_dir"] = f"""./model/{config["orig_source_length"]}_{config["max_source_length"]}_{config["method"]}"""
config["run_name"] = f"""{config["orig_source_length"]}_{config["max_source_length"]}_{config["method"]}"""
