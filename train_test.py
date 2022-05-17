# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
import os
from datasets import load_metric
from utils import *

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console
from IPython.display import clear_output

# Importing the T5 modules from huggingface/transformers
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, LEDTokenizer, LEDForConditionalGeneration

import nltk

from dataset import Dataset

console = Console(record=True)

training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)


def train(epoch, tokenizer, model, device, loader, optimizer, scheduler, model_params):
    
    """
    Function to be called for training with the parameters passed from trainer function

    """

    model.train()
    losses = 0
    results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Reference summary": [], "Generated summary": [], "Document length": [], "Shortened document length": [], "Reference length": [], "Generated length": [] } 
    for _, data in enumerate(loader, 0):
        if model_params["METHOD"] in ["full-text", "head-only", "tail-only"]+["head+tail_ratio{:.1f}".format(i) for i in np.arange(0.0, 1.0, 0.1)]:
            len_check1 = data["source_len"] <= 512 
            len_check2 = data["source_len"] >= 485
            len_check3 = data["target_len"] <= 36
            len_check = len_check1 & len_check2 & len_check3
            if model_params["RESTRICTION"] == True:
                if len_check.sum() < len(data["source_len"]):
                    print("SOSSSSSS")
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 1000 == 0:
            print("STEP: ", _,"/",len(loader))
            training_logger.add_row(str(epoch), str(f'{_}/{len(loader)}'), str(loss))
            console.print(training_logger)
            clear_output(wait=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.detach()
        preds, target, preds_len = generate(ids, mask, model, tokenizer)
        results = prepare_results(results, data, preds, target, preds_len)

    scheduler.step()
    losses = losses/len(loader)
    rouge = compute_metrics(results, tokenizer)
    return losses, rouge

def validate(epoch, tokenizer, model, device, loader):

    """
    Function to evaluate model for predictions

    """
    model.eval()
    losses = 0
    results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Reference summary": [], "Generated summary": [], "Document length": [], "Shortened document length": [], "Reference length": [], "Generated length": [] } 
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100     
            
            outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
                )
            loss = outputs[0]
            losses += loss.detach()
            
            preds, target, preds_len = generate(ids, mask, model, tokenizer)
            results = prepare_results(results, data, preds, target, preds_len)
                
    losses = losses/len(loader)
    rouge = compute_metrics(results, tokenizer)
    return losses, rouge

def inference(best_checkpoint_path, tokenizer, model, device, loader):
    """
    Function to evaluate model for test predictions

    """
    model.load_state_dict(torch.load(os.path.join(best_checkpoint_path, 'pytorch_model.bin'), map_location="cpu")) 
    tokenizer.from_pretrained(best_checkpoint_path)
    model.eval()
    results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Reference summary": [], "Generated summary": [], "Document length": [], "Shortened document length": [], "Reference length": [], "Generated length": [] } 
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            preds, target, preds_len = generate(ids, mask, model, tokenizer)
            
            if _ % 1000==0:
                console.print(f'Completed {_}')
            results = prepare_results(results, data, preds, target, preds_len)
    rouge = compute_metrics(results, tokenizer)
    return results


def generate(ids, mask, model, tokenizer):
    generated_ids = model.generate(
                  input_ids = ids,
                  attention_mask = mask, 
                  max_length = 36, 
                  num_beams=2,
                  repetition_penalty=2.5, 
                  length_penalty=1.0, 
                  early_stopping=True
                  )
             
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
    preds_len = [len(tokenizer.batch_encode_plus([p],return_tensors="pt")["input_ids"].squeeze()) for p in preds]
    return preds, target, preds_len

def prepare_results(results, data, preds, target, preds_len):
    results['Sample ids'].extend(data['ids'])
    results['Document'].extend(data['source_text'])
    results['Shortened Document'].extend(data['shortened_source_text'])
    results['Reference summary'].extend(target)
    results['Generated summary'].extend(preds)
    results['Document length'].extend(data['source_len'].tolist())
    results['Shortened document length'].extend(data['shortened_source_len'].tolist())
    results['Reference length'].extend(data['target_len'].tolist())
    results['Generated length'].extend(preds_len)
    return results
    
def Trainer(
    dataset, source_text, target_text, model_params, output_dir="./outputs/", device = "cuda", mask = None, to_mask_list = None,
):

    """
    trainer funcation defines Tokenizer, creates Dataset, Dataloaders according to the shortening strategy

    """
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # tokenzier for encoding the text
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    if "bart" in model_params["MODEL"]:
        tokenizer = BartTokenizer.from_pretrained(f'facebook/{model_params["MODEL"]}')
        model = BartForConditionalGeneration.from_pretrained(f'facebook/{model_params["MODEL"]}')
    elif "t5" in model_params["MODEL"]:
        tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
        model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    elif "pegasus" in model_params["MODEL"]:
        tokenizer = PegasusTokenizer.from_pretrained(f'google/{model_params["MODEL"]}')
        model = PegasusForConditionalGeneration.from_pretrained(f'google/{model_params["MODEL"]}')
    elif "longformer" in model_params["MODEL"]:
        tokenizer = LEDTokenizer.from_pretrained(f'allenai/{model_params["MODEL"]}') #longformer-base-4096
        model = LEDForConditionalGeneration.from_pretrained(f'allenai/{model_params["MODEL"]}')        
    else:
        raise ValueError("Undefined model")
        

    if isinstance(model_params["RESUME_FROM_CHECKPOINTS"], bool) and model_params["RESUME_FROM_CHECKPOINTS"]:
        resume_checkpoint_path,resumed_epoch  = get_last_checkpoint(os.path.join(output_dir, f"""checkpoints"""))
        model.load_state_dict(torch.load(os.path.join(resume_checkpoint_path, 'pytorch_model.bin'), map_location="cpu")) 
        tokenizer.from_pretrained(resume_checkpoint_path)
        model_params["START_TRAIN_EPOCHS"] = resumed_epoch
        train_losses = list(np.load(os.path.join(resume_checkpoint_path, f"""train_losses_{model_params['MODEL']}_epoch{resumed_epoch}.npy""")))
        val_losses = list(np.load(os.path.join(resume_checkpoint_path, f"""val_losses_{model_params['MODEL']}_epoch{resumed_epoch}.npy""")))
        print(f"""[Model, Tokenizer]: Resuming at epoch{model_params["START_TRAIN_EPOCHS"]}...\n""")
    else: 
        model_params["START_TRAIN_EPOCHS"] = 0
        train_losses = []
        val_losses = []
        
    model = model.to(device)
    
    # logging
    console.log(f"[Data]: Reading data...\n")
    
    if model_params["METHOD"] in ["full-text", "head-only", "tail-only",]+["head+tail_ratio{:.1f}".format(i) for i in np.arange(0.0, 1.0, 0.1)]:

        # Creation of Dataset and Dataloader for full-text, head-only, tail-only and head+tail truncation
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        test_dataset = dataset["test"]

        console.print(f"FULL Dataset: {dataset.shape}")
        console.print(f"TRAIN Dataset: {train_dataset.shape}")
        console.print(f"TEST Dataset: {val_dataset.shape}\n")

        path_train = "datalength/train_info.csv"
        df_train = pd.read_csv(path_train)
        
        path_val = "datalength/val_info.csv"
        df_val = pd.read_csv(path_val) 

        path_test = "datalength/test_info.csv"
        df_test = pd.read_csv(path_test)    

        if model_params["RESTRICTION"] == True:
            print("LEN RESTRICTION")
            # less than n input tokens

            traincond1 = df_train["doc len"] >= 485 # 485 972
            traincond2 = df_train["doc len"] <= 512 #512 1024
            traincond3 = df_train["sum len"] <= 36
            
            valcond1 = df_val["doc len"] >= 485 # 485 972
            valcond2 = df_val["doc len"] <= 512 #512 1024
            valcond3 = df_val["sum len"] <= 36

            testcond1 = df_test["doc len"] >= 485 # 485 972
            testcond2 = df_test["doc len"] <= 512 #512 1024
            testcond3 = df_test["sum len"] <= 36

            index_train = df_train['index'][traincond1 & traincond2 & traincond3]
            index_val = df_val['index'][valcond1 & valcond2 & valcond3]
            index_test = df_test['index'][testcond1 & testcond2 & testcond3]

        else:
            print("NO LEN RESTRICTION")
            index_train = df_train['index']
            index_val = df_val['index']
            index_test = df_test['index']

        # Shuffle
        index_train = np.random.permutation(index_train)
        index_val = np.random.permutation(index_val)
        index_test = np.random.permutation(index_test)

        console.print(f"Filtered TRAIN Dataset: {len(train_dataset[index_train]['id'])}")
        console.print(f"Filtered VAL Dataset: {len(val_dataset[index_val]['id'])}\n")
        console.print(f"Filtered TEST Dataset: {len(test_dataset[index_test]['id'])}\n")
        
        train_dataset_ = train_dataset[index_train]
        val_dataset_ = val_dataset[index_val]
        test_dataset_ = test_dataset[index_test]
        
    else:
        
        # Creation of Dataset and Dataloader for preprocessed text e.g. luhn, lsa textrank....
        train_dataset_ = pd.read_csv(os.path.join(model_params["PATH"], f"""train_set.csv"""))
        val_dataset_ = pd.read_csv(os.path.join(model_params["PATH"], f"""val_set.csv"""))
        test_dataset_ = pd.read_csv(os.path.join(model_params["PATH"], f"""test_set.csv"""))
    
    console.print(f"""GOT {model_params["METHOD"]} PREPROCESSED TEXT""")
    
    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = Dataset(
        train_dataset_,
        tokenizer,
        model_params["MODEL"],
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
        model_params["METHOD"],
    )
    
    val_set = Dataset(
        val_dataset_,
        tokenizer,
        model_params["MODEL"],
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
        model_params["METHOD"]
    )
    
    test_set = Dataset(
        test_dataset_,
        tokenizer,
        model_params["MODEL"],
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
        model_params["METHOD"]
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }
    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }
    test_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)
    
    console.print("TRAIN LOADER: ", len(training_loader))
    console.print("VAL LOADER: ", len(val_loader))
    console.print("TEST LOADER: ", len(test_loader))
    
    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )
    
    if model_params["SCHEDULER"] == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")
    
    best_model_loss = 10000000
    for epoch in range(model_params["START_TRAIN_EPOCHS"], model_params["TRAIN_EPOCHS"]):
        console.print(f"[Start training epoch {epoch}]...")
        train_loss, train_result = train(epoch, tokenizer, model, device, training_loader, optimizer, scheduler, model_params)
        train_losses.append(train_loss.cpu().numpy())
        train_rouge1s.append(train_results['rouge1'])
        train_rouge2s.append(train_results['rouge2'])
        train_rougeLs.append(train_results['rougeL'])
        train_rougeLsums.append(train_results['rougeLsum'])
        train_rougelen.append(train_results['gen len'])
        print(train_losses)
        console.print(f"[Start validating epoch {epoch}]...")
        # evaluating val dataset        
        val_loss, val_result = validate(epoch, tokenizer, model, device, val_loader)
        val_losses.append(val_loss.cpu().numpy())
        print(val_losses)
        
        if not os.path.exists(output_dir):
            print(f"{output_dir} CREATED!")
            os.makedirs(output_dir)
            os.makedirs(os.path.join(output_dir, f"""result_gen"""))
            os.makedirs(os.path.join(output_dir, f"""result_eval"""))
            os.makedirs(os.path.join(output_dir, f"""checkpoints"""))
            
        # Saving the best model after training
        # Converting list to array
        if val_loss < best_model_loss:
            path_best = os.path.join(output_dir, f"checkpoints/epoch{epoch}")
            model.save_pretrained(path_best)
            tokenizer.save_pretrained(path_best)
            train_losses_arr = np.array(train_losses)
            np.save(os.path.join(output_dir, f"""checkpoints/epoch{epoch}/train_losses_{model_params['MODEL']}_epoch{epoch}"""), train_losses_arr)
            val_losses_arr = np.array(val_losses)
            np.save(os.path.join(output_dir, f"""checkpoints/epoch{epoch}/val_losses_{model_params['MODEL']}_epoch{epoch}"""), val_losses_arr)
            best_model_loss = val_loss
    # Inference with best model
    best_checkpoint_path  = get_best_checkpoint(os.path.join(output_dir, f"""checkpoints"""))
    results = inference(best_checkpoint_path , tokenizer, model, device, test_loader)
    
    final_df = pd.DataFrame(results)
    final_df.to_csv(os.path.join(output_dir, f"""result_gen/predictions_{model_params['MODEL']}.csv"""))
    final_df.to_csv(os.path.join(output_dir, f"""result_eval/predictions_{model_params['MODEL']}.csv"""))
    print("SAVE TO CSV FINISHED")

    rouge = compute_metrics(results["Generated summary"], results["Reference summary"], tokenizer)

    rouge_df = pd.DataFrame.from_dict(rouge, orient='index')
    rouge_df.to_csv(os.path.join(output_dir, f"""result_eval/rouge_{model_params['MODEL']}.csv"""))
    print("SAVE ROUGE TO CSV FINISHED")
#     if not os.path.exists(os.path.join(output_dir, f"""checkpoints/epoch{epoch}""")):
#         os.makedirs(os.path.join(output_dir, f"""checkpoints/epoch{epoch}"""))
#     console.log(f"[Saving Model at EPOCH {epoch}]...\n")

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")
    

def compute_metrics(results, tokenizer):

    metric = load_metric("rouge")
    result = metric.compute(predictions=results['Generated summary'], references=results["Reference summary"], use_stemmer=True)
    
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    result["gen_len"] = np.mean(results["Generated length"])
    hi = {k: round(v, 4) for k, v in result.items()}
    return hi