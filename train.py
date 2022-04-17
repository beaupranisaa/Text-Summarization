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


def train(epoch, tokenizer, model, device, loader, optimizer, scheduler, len_restriction = False):
    
    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    losses = 0
    for _, data in enumerate(loader, 0):
        len_check1 = data["source_len"] <= 512 
        len_check2 = data["source_len"] >= 485
        len_check3 = data["target_len"] <= 36
        len_check = len_check1 & len_check2 & len_check3
#         len_check.sum()
        if len_restriction == True:
            if len_check.sum() < len(data["source_len"]):
                print("SOSSSSSS")
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
#         print("SHAPE IDS: ", ids.shape)
#         print("SHAPE y: ", y.shape)
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
    # #             print("TRAINNNNNNN")
    # #             print(epoch)
            training_logger.add_row(str(epoch), str(f'{_}/{len(loader)}'), str(loss))
            console.print(training_logger)
            clear_output(wait=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.detach()

#         del _, data, y, y_ids, lm_labels, ids, mask, outputs, loss
    scheduler.step()
    losses = losses/len(loader)
#     del loader 
    return losses

def validate(epoch, tokenizer, model, device, loader):

    """
    Function to evaluate model for predictions

    """
    model.eval()
#     losses = 0
    results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Reference summary": [], "Generated summary": [], "Document length": [], "Reference length": [], "Generated length": [] }
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

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
            
            if _ % 1000==0:
                console.print(f'Completed {_}')
                
            results['Sample ids'].extend(data['ids'])
            results['Document'].extend(data['source_text'])
            results['Shortened Document'].extend(data['shortened_source_text'])
            results['Reference summary'].extend(target)
            results['Generated summary'].extend(preds)
            results['Document length'].extend(data['source_len'].tolist())
            results['Reference length'].extend(data['target_len'].tolist())
            results['Generated length'].extend(preds_len)

    del loader
    return results

def Trainer(
    dataset, source_text, target_text, model_params, output_dir="./outputs/", device = "cuda", len_restriction = False, mask = None, to_mask_list = None
):

    """
    trainer

    """
    
    losses = []
    losses_val = []
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
        
    model = model.to(device)
    print(model)
    # logging
    console.log(f"[Data]: Reading data...\n")

    # Creation of Dataset and Dataloader
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    
    console.print(f"FULL Dataset: {dataset.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    path_train = "datalength/train_info.csv"
    df_train = pd.read_csv(path_train)
    
    path_test = "datalength/test_info.csv"
    df_test = pd.read_csv(path_test)    
    
    if len_restriction == True:
        print("LEN RESTRICTION")
        # less than n input tokens
        
        traincond1 = df_train["doc len"] >= 485
        traincond2 = df_train["doc len"] <= 512
        traincond3 = df_train["sum len"] <= 36
        
        testcond1 = df_test["doc len"] >= 485
        testcond2 = df_test["doc len"] <= 512
        testcond3 = df_test["sum len"] <= 36
        
        index_train = df_train['index'][traincond1 & traincond2 & traincond3]
        index_test = df_test['index'][testcond1 & testcond2 & testcond3]
        print(len(index_train))
        print(len(index_test))
        
    else:
        print("NO LEN RESTRICTION")
        index_train = df_train['index']
        index_test = df_test['index']
        
    # Shuffle and take only 1000 samples
    index_train = np.random.permutation(index_train)
    index_test = np.random.permutation(index_test)
#     print(index_train)
        
    console.print(f"Filtered TRAIN Dataset: {len(train_dataset[index_train]['id'])}")
    console.print(f"Filtered TEST Dataset: {len(test_dataset[index_test]['id'])}\n")

    del dataset

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = Dataset(
        train_dataset[index_train],
        tokenizer,
        model_params["MODEL"],
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
        mask = mask,
        to_mask_list = to_mask_list,
    )
    test_set = Dataset(
        test_dataset[index_test],
        tokenizer,
        model_params["MODEL"],
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    del train_dataset, test_dataset
    
    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
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
    test_loader = DataLoader(test_set, **test_params)
    
    print("TRAIN LOADER: ", len(training_loader))
    print("VAL LOADER: ", len(test_loader))
    
    del train_params, test_params
    
    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )
    
    if model_params["SCHEDULER"] == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        print("TRAIN")
        loss = train(epoch, tokenizer, model, device, training_loader, optimizer, scheduler,  len_restriction = len_restriction)
#         break
        losses.append(loss.cpu().numpy())
        print("VALIDATE")
        # evaluating test dataset        
        results = validate(epoch, tokenizer, model, device, test_loader)
#         losses_val.append(loss_val.cpu().numpy())
        
        if not os.path.exists(output_dir):
            print(f"{output_dir} CREATED!")
            os.makedirs(output_dir)
            os.makedirs(os.path.join(output_dir, f"""result_gen"""))
            os.makedirs(os.path.join(output_dir, f"""result_eval"""))
            os.makedirs(os.path.join(output_dir, f"""model_files"""))
        
#         final_df = pd.DataFrame({"ids": ids, "Document": documents, "Generated Text": predictions, "Actual Text": actuals, "Document length": document_lens, "Actual length": actuals_lens })
        
        final_df = pd.DataFrame(results)
        final_df.to_csv(os.path.join(output_dir, f"""result_gen/predictions_{model_params['MODEL']}_epoch{epoch}.csv"""))
        final_df.to_csv(os.path.join(output_dir, f"""result_eval/predictions_{model_params['MODEL']}_epoch{epoch}.csv"""))
        print("SAVE TO CSV FINISHED")
        
        rouge = compute_metrics(results["Generated summary"], results["Reference summary"], tokenizer)
        
        rouge_df = pd.DataFrame.from_dict(rouge, orient='index')
        rouge_df.to_csv(os.path.join(output_dir, f"""result_eval/rouge_{model_params['MODEL']}_epoch{epoch}.csv"""))
        print("SAVE ROUGE TO CSV FINISHED")
#         del  loss, predictions, actuals, final_df, rouge, rouge_df #losses_val,
    
    del training_loader, test_loader
    
    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    
    # converting list to array
    arr = np.array(losses)
    np.save(os.path.join(output_dir, f"""losses_{model_params['MODEL']}_epoch{model_params['TRAIN_EPOCHS']}"""), arr)
    arr_val = np.array(losses_val)
    np.save(os.path.join(output_dir, f"""losses_val_{model_params['MODEL']}_epoch{model_params['VAL_EPOCHS']}"""), arr_val)

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")
    

def compute_metrics(predictions, actuals, tokenizer):

    metric = load_metric("rouge")
    result = metric.compute(predictions=predictions, references=actuals, use_stemmer=True)
    
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    hi = {k: round(v, 4) for k, v in result.items()}
    del metric, result, prediction_lens
    return hi