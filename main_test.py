# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import pickle
# from train import Trainer
from train_test import Trainer
from utils import *
from torch import cuda
import os
from datasets import load_dataset
from config_test import *
from tqdm.notebook import tqdm

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

import nltk

from dataset import Dataset

import wandb
from datasets import load_metric
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

def model_pipeline(config):
    device = 'cuda' if cuda.is_available() else 'cpu'
    print("configured device: ", device)
    
    torch.manual_seed(config['seed'])  # pytorch random seed
    np.random.seed(config['seed'])  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    # tell wandb to get started
    with wandb.init(project = "textsummarization",
                    name = config["run_name"],
                    config = config,
                    resume = config["resume_from_checkpoint"],
                    dir = config["output_dir"],):
        
        config = wandb.config

        result_table, result_dict, summary_dict = prepare_test_output()
        
        # make the model, data, and optimization problem
        model, tokenizer, train_loader, val_loader, test_loader, optimizer, scheduler, start_epoch, train_losses, val_losses = make(config, device)

        # and use them to train the model
        train(model, tokenizer, train_loader, val_loader, optimizer, scheduler, config, start_epoch, train_losses, val_losses,  device)

        # and test its final performance
        test(tokenizer, model, device, test_loader, result_table, result_dict, summary_dict, config)

        return model

def make(config, device):
    
    tokenizer = T5Tokenizer.from_pretrained(config.model)
    model = T5ForConditionalGeneration.from_pretrained(config.model)
    
     # Make the tokenizer and model
    if config.resume_from_checkpoint == True:
        start_epoch  = get_last_checkpoint(os.path.join(config.output_dir, f"""checkpoints/current_model"""))
        model.load_state_dict(torch.load(os.path.join(config.output_dir, 'checkpoints/current_model/pytorch_model.bin'), map_location="cpu")) 
        tokenizer.from_pretrained(os.path.join(config.output_dir, 'checkpoints/current_model'))
        train_losses = list(np.load(os.path.join(config.output_dir, f"""checkpoints/current_model/train_losses.npy""")))
        val_losses = list(np.load(os.path.join(config.output_dir, f"""checkpoints/current_model/val_losses.npy""")))
    else:
        start_epoch  = 0
        train_losses, val_losses = [], []
    model = model.to(device)
    
    # Make the data
    train, val, test = get_data(config, tokenizer, mode = "train"), get_data(config, tokenizer, mode = "val"), get_data(config, tokenizer, mode = "test")
    
    train_loader, val_loader, test_loader = make_loader(train, config), make_loader(val, config), make_loader(test, config)
    print("TRAIN LOADER: ", len(train_loader))
    print("VAL LOADER: ", len(val_loader))
    print("TEST LOADER: ", len(test_loader))
    # Make the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    return model, tokenizer, train_loader, val_loader, test_loader, optimizer, scheduler, start_epoch, train_losses, val_losses

def get_data(config, tokenizer, mode = "train", source_text_col = "document", target_text_col = "summary"):
    
    data = pd.read_csv(os.path.join(config.path, f"""{mode}_set.csv"""))[:10]
    
    dataset = Dataset(
        data,
        tokenizer,
        config.model,
        config.max_source_length,
        config.max_target_length,
        source_text_col,
        target_text_col,
        config.method
    )

    return dataset


def make_loader(dataset, config):
    loader_params = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": 0,
    }
    loader = DataLoader(dataset, **loader_params)
    
    return loader

def train(model, tokenizer, train_loader, val_loader, optimizer, scheduler, config, start_epoch, train_losses, val_losses, device):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, log="all", log_freq=10)
    
    prev_loss = 1000000
    # Run training and track with wandb
    for epoch in tqdm(range(start_epoch, config.train_epochs)):
        print(f"AT EPOCH {epoch}")
        train_loss = train_epoch(epoch, model, tokenizer, train_loader, optimizer, scheduler, device)
        val_loss = val_epoch(epoch, model, tokenizer, val_loader, device)
        train_losses.append(train_loss.cpu().numpy())
        val_losses.append(val_loss.cpu().numpy())
        write_log(train_loss, val_loss)
        
        if not os.path.exists(config.output_dir):
            print(f"{config.output_dir} CREATED!")
            os.makedirs(config.output_dir)
            os.makedirs(os.path.join(config.output_dir, f"""checkpoints"""))
        
        save_current(model, tokenizer, train_losses, val_losses, epoch, config)
        
        if val_loss < prev_loss:
            save_best(model, tokenizer, train_loss.cpu().numpy(), val_loss.cpu().numpy(), epoch, config)
            prev_loss = val_loss
        
def save_current(model, tokenizer, train_losses, val_losses, epoch, config):
    path_current = os.path.join(config.output_dir, f"checkpoints/current_model")
    model.save_pretrained(path_current)
    tokenizer.save_pretrained(path_current)
    train_losses_arr = np.array(train_losses)
    val_losses_arr = np.array(val_losses)
    np.save(f"{path_current}/train_losses.npy", train_losses_arr)
    np.save(f"{path_current}/val_losses.npy", val_losses)
    np.save(f"{path_current}/current_epoch.npy", epoch)
        
def save_best(model, tokenizer, train_loss, val_loss, epoch, config):
    path_best = os.path.join(config.output_dir, f"checkpoints/best_model")
    model.save_pretrained(path_best)
    tokenizer.save_pretrained(path_best)
    np.save(f"{path_best}/best_train_loss.npy", train_loss)
    np.save(f"{path_best}/best_val_loss.npy", val_loss)
    np.save(f"{path_best}/best_epoch.npy", epoch)

def train_epoch(epoch, model, tokenizer, loader, optimizer, scheduler, device):
    model.train()
    losses = 0
    print("TRAINING")
    for _, data  in enumerate(loader, 0):
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
        
        # Backward pass ⬅
        optimizer.zero_grad()
        loss.backward()
        # Step with optimizer
        optimizer.step()  
        
        losses += loss.detach()
    
    scheduler.step()
    losses = losses/len(loader)
    print(f"TRAIN LOSS at {epoch}: {losses}")
    return losses

def val_epoch(epoch, model, tokenizer, loader, device):

    """
    Function to evaluate model for predictions

    """
    model.eval()
    losses = 0
    print("VALIDATING")
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
                
    losses = losses/len(loader)
    print(f"VAL LOSS at {epoch}: {losses}")
    return losses
    
def test(tokenizer, model, device, loader, result_table, result_dict, summary_dict, config):
    """
    Function to evaluate model for test predictions

    """
    best_checkpoint_path = os.path.join(config.output_dir, f"checkpoints/best_model")
    model.load_state_dict(torch.load(os.path.join(best_checkpoint_path, 'pytorch_model.bin'), map_location="cpu"))
    tokenizer.from_pretrained(best_checkpoint_path)
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            pred, target, pred_len = generate(y, ids, mask, model, tokenizer)
            prepared_result_dict = prepare_results(result_dict, data, pred, target, pred_len)
            
    save_results_to_csv(prepared_result_dict, config)
    rouges = compute_metrics(prepared_result_dict, 'rouge')
    write_summary(summary_dict, rouges, config)
    
def generate(y, ids, mask, model, tokenizer):
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
    
def prepare_test_output():
    result_columns = ["Sample id", "Document", "Shortened Document", 
               "Reference summary", "Generated summary",
               "Document length", "Shortened document length",
               "Reference length", "Generated length",]
    result_table = wandb.Table(columns=result_columns)
    result_dict = {key: [] for key in result_columns}

    summary_columns = ["test rouge1", "test rouge2", "test rougeL", "test rougeLsum", "test gen len", "best epoch"]
    summary_dict = {key: [] for key in summary_columns}
    
    return result_table, result_dict, summary_dict

def write_log(train_loss, val_loss):
    # Where the magic happens
    wandb.log({"train loss": train_loss, "val loss": val_loss,})
    print("LOG WRITTEN")

def write_summary(summary_dict, rouges, config):
    
    best_epoch = int(np.load(os.path.join(config.output_dir, f"checkpoints/best_model/best_epoch.npy")))
    best_epoch_train_loss = np.load(os.path.join(config.output_dir, f"checkpoints/best_model/best_train_loss.npy"))
    best_epoch_val_loss = np.load(os.path.join(config.output_dir, f"checkpoints/best_model/best_val_loss.npy"))
    
    wandb.summary["test rouge1"] = summary_dict["test rouge1"] = rouges['rouge1']
    wandb.summary["test rouge2"] = summary_dict["test rouge2"] = rouges['rouge2']
    wandb.summary["test rougeL"] = summary_dict["test rougeL"] = rouges['rougeL']
    wandb.summary["test rougeLsum"] = summary_dict["test rougeLsum"] = rouges['rougeLsum']
    wandb.summary["test gen len"] = summary_dict["test gen len"] = rouges['gen_len']
    wandb.summary["best epoch"] = summary_dict["best epoch"] = best_epoch
    wandb.summary["best epoch train loss"] = summary_dict["best epoch train loss"] = best_epoch_train_loss
    wandb.summary["best epoch val loss"] = summary_dict["best epoch val loss"] = best_epoch_val_loss
    # create a binary pickle file 
    f = open(os.path.join(config.output_dir, f"checkpoints/best_model/summary_dict.pkl"),"wb")
    # write the python object (dict) to pickle file
    pickle.dump(summary_dict,f)
    # close file
    f.close()

def save_results_to_csv(prepared_result_dict, config):
    result_df = pd.DataFrame(prepared_result_dict)
    result_df.to_csv(os.path.join(config.output_dir, f"checkpoints/best_model/prediction.csv"))
    
def prepare_results(results, data, preds, target, preds_len):
    results['Sample id'].extend(data['ids'].tolist())
    results['Document'].extend(data['source_text'])
    results['Shortened Document'].extend(data['shortened_source_text'])
    results['Reference summary'].extend(target)
    results['Generated summary'].extend(preds)
    results['Document length'].extend(data['source_len'].tolist())
    results['Shortened document length'].extend(data['shortened_source_len'].tolist())
    results['Reference length'].extend(data['target_len'].tolist())
    results['Generated length'].extend(preds_len)
    return results

def compute_metrics(result_dict, score = 'rouge'):
    if score == 'rouge':
        metric = load_metric("rouge")
        result = metric.compute(predictions=result_dict['Generated summary'], references=result_dict["Reference summary"], use_stemmer=True)

        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        result["gen_len"] = np.mean(result_dict["Generated length"])
#         result["gen_len"] = np.mean(preds_len)
        rougescore = {k: round(v, 4) for k, v in result.items()}
        return rougescore
    elif score == 'bertscore':
        metric = load_metric("bertscore")
        bertscore = metric.compute(predictions=result_dict['Generated summary'], references=result_dict["Reference summary"], rescale_with_baseline = True, lang = 'en')
        return bertscore
    else:
        raise ValueError("Undefined metric...")


os.environ["WANDB_API_KEY"] = '82391ac94007d5b4aa987d46308aa30e26a4b794'        
os.environ["WANDB_MODE"] = "offline"
wandb.login()            
model = model_pipeline(config)
wandb.finish()