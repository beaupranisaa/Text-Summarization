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

        columns = ["Sample id", "Document", "Shortened Document", 
                   "Reference summary", "Generated summary",
                   "Document length", "Shortened document length",
                   "Reference length", "Generated length",]
        
        result_table = wandb.Table(columns=columns)

        # make the model, data, and optimization problem
        model, tokenizer, train_loader, val_loader, test_loader, optimizer, scheduler, start_epoch = make(config, device)

        # and use them to train the model
        train(model, tokenizer, train_loader, val_loader, optimizer, scheduler, config, start_epoch, device)

        # and test its final performance
        test(tokenizer, model, device, test_loader, result_table, config)

        return model

def make(config, device):
    
    tokenizer = T5Tokenizer.from_pretrained(config.model)
    model = T5ForConditionalGeneration.from_pretrained(config.model)
    
     # Make the tokenizer and model
    if config.resume_from_checkpoint == True:
        start_epoch  = get_last_checkpoint(os.path.join(config.output_dir, f"""checkpoints/current_model"""))
        model.load_state_dict(torch.load(os.path.join(config.output_dir, 'checkpoints/current_model/pytorch_model.bin'), map_location="cpu")) 
        tokenizer.from_pretrained(os.path.join(config.output_dir, 'checkpoints/current_model'))
    else:
        start_epoch  = 0
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
    
    return model, tokenizer, train_loader, val_loader, test_loader, optimizer, scheduler, start_epoch

def get_data(config, tokenizer, mode = "train", source_text_col = "document", target_text_col = "summary"):
    
    data = pd.read_csv(os.path.join(config.path, f"""{mode}_set.csv"""))
    
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

def train(model, tokenizer, train_loader, val_loader, optimizer, scheduler, config, start_epoch, device):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, log="all", log_freq=10)
    
    prev_loss = 1000000
    # Run training and track with wandb
    for epoch in tqdm(range(start_epoch, config.train_epochs)):
        print(f"AT EPOCH {epoch}")
        train_loss = train_epoch(epoch, model, tokenizer, train_loader, optimizer, scheduler, device)
        val_loss = val_epoch(epoch, model, tokenizer, val_loader, device)
        write_log(train_loss, val_loss)
        if not os.path.exists(config.output_dir):
            print(f"{config.output_dir} CREATED!")
            os.makedirs(config.output_dir)
            os.makedirs(os.path.join(config.output_dir, f"""checkpoints"""))
        
        path_current = os.path.join(config.output_dir, f"checkpoints/current_model")
        model.save_pretrained(path_current)
        tokenizer.save_pretrained(path_current)
        np.save(f"{path_current}/current_epoch.npy", epoch)
        
        if val_loss < prev_loss:
            path_best = os.path.join(config.output_dir, f"checkpoints/best_model")
            model.save_pretrained(path_best)
            tokenizer.save_pretrained(path_best)
            np.save(f"{path_best}/best_epoch.npy", epoch)
            prev_loss = val_loss
        
                
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
    
def test(tokenizer, model, device, loader, result_table, config):
    """
    Function to evaluate model for test predictions

    """
    best_checkpoint_path = os.path.join(config.output_dir, f"checkpoints/best_model")
    best_epoch = int(np.load(os.path.join(config.output_dir, f"checkpoints/best_model/best_epoch.npy")))
    model.load_state_dict(torch.load(os.path.join(best_checkpoint_path, 'pytorch_model.bin'), map_location="cpu")) 
    tokenizer.from_pretrained(best_checkpoint_path)
    model.eval()
    preds, targets, pred_lens= [], [], []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            pred, target, pred_len = generate(y, ids, mask, model, tokenizer)
            preds.extend(pred)
            targets.extend(target)
            pred_lens.extend(pred_len)
            
            for i in range(len(data['ids'])):
                result_table.add_data(data['ids'][i], data['source_text'][i], 
                                       data['shortened_source_text'][i],
                                       target[i], pred[i], data['source_len'].tolist()[i], 
                                       data['shortened_source_len'].tolist()[i],
                                       data['target_len'].tolist()[i], pred_len[i],)
    
    rouges = compute_metrics(preds, targets, pred_lens, 'rouge')
    wandb.log({"test results" : result_table})
    wandb.summary["test rouge1"] = rouges['rouge1']
    wandb.summary["test rouge2"] = rouges['rouge2']
    wandb.summary["test rougeL"] = rouges['rougeL']
    wandb.summary["test rougeLsum"] = rouges['rougeLsum']
    wandb.summary["test gen len"] = rouges['gen_len']
    wandb.summary["best epoch"] = best_epoch
#     wandb.log{"rouge score"}

   
    
def write_log(train_loss, val_loss):
    # Where the magic happens
    wandb.log({"train loss": train_loss, "val loss": val_loss,})
    print("LOG WRITTEN")
    
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

def compute_metrics(preds, target, preds_len, score = 'rouge'):
    if score == 'rouge':
        metric = load_metric("rouge")
        result = metric.compute(predictions=preds, references=target, use_stemmer=True)

        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        result["gen_len"] = np.mean(preds_len)
        rougescore = {k: round(v, 4) for k, v in result.items()}
        return rougescore
    elif score == 'bertscore':
        metric = load_metric("bertscore")
        bertscore = metric.compute(predictions=preds, references=target, rescale_with_baseline = True, lang = 'en')
        return bertscore
    else:
        raise ValueError("Undefined metric...")


os.environ["WANDB_API_KEY"] = '82391ac94007d5b4aa987d46308aa30e26a4b794'        
os.environ["WANDB_MODE"] = "offline"
wandb.login()            
model = model_pipeline(config)
wandb.finish()