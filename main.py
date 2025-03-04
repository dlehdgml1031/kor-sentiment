import os
import re
import datasets

import torch
import wandb
import pandas as pd
import numpy as np
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def _preporcess_text(text):
    text = text.replace(
        '\n', '').replace(
        '\t', '').replace(
        '\r', '').replace(
        "\'", "").replace(
        '-', '').replace(
        '.', '').replace(
        ',', '').replace(
        '?', '').replace(
        '!', '').replace(
        '  ', ' ').replace(
        '   ', ' ').strip()
    
    return ' '.join(text.split())


def load_data(
    train_file_path:str,
    test_file_path:str,
):
    train_df = pd.read_csv(train_file_path, sep = '\t')
    test_df = pd.read_csv(test_file_path, sep = '\t')
    
    train_df = train_df.sample(frac = 1, random_state = 42).reset_index(drop = True)
    test_df = test_df.sample(frac = 1, random_state = 42).reset_index(drop = True)
    
    # train_df['text'] = train_df['text'].apply(_preporcess_text)
    # test_df['text'] = test_df['text'].apply(_preporcess_text)
    
    train_dataset = datasets.Dataset.from_dict({'text' : train_df['text'].to_list(), 'label' : train_df['label'].to_list()})
    test_dataset = datasets.Dataset.from_dict({'text' : test_df['text'].to_list(), 'label' : test_df['label'].to_list()})
    
    train_dataset = train_dataset.class_encode_column("label")
    test_dataset = test_dataset.class_encode_column("label")
    
    # split train test with seed 0
    return datasets.DatasetDict(
        {
            'train' : train_dataset,
            'test' : test_dataset,
        }
    )


def train_lmkor(
    base_model_name:str,
    dataset_name:str,
    batch_size:int,
    learning_rate:float,
    epoch:int,
):
    # # login wandb
    # wandb.login()
    # os.environ['WANDB_PROJECT'] = 'Kor Sentiment'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load base model
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, max_length = 64)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels = 2).to(device)
    
    
    # load dataset
    data = load_data(
        train_file_path = f'./data/50000/{dataset_name}_train.tsv',
        test_file_path = f'./data/50000/{dataset_name}_test.tsv',
    )
    
    # tokenized text
    tokenized_data = data.map(lambda examples: base_tokenizer(examples['text'], truncation = True), batched = True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer = base_tokenizer)
    
    # netsted function for compute metrics
    def _compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = 'binary')
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, preds)
        return {
            "accuracy" : acc,
            "f1" : f1,
            "precision" : precision,
            "recall" : recall,
            'auroc' : auc
        }
    
    # define training arguments
    # using AdamW optimizer
    training_args = transformers.TrainingArguments(
        output_dir = f'./results/lmkor_{dataset_name}',
        run_name = f'lmkor_{base_model_name}_{dataset_name}',
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        learning_rate = learning_rate,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        num_train_epochs = epoch,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        report_to = 'wandb',
        logging_dir = './logs',
        logging_steps = 50,
        seed = 42
    )
    
    trainer = transformers.Trainer(
        model = base_model,
        tokenizer = base_tokenizer,
        args = training_args,
        train_dataset = tokenized_data['train'],
        eval_dataset = tokenized_data['test'],
        data_collator = data_collator,
        compute_metrics = _compute_metrics,
    )
    
    trainer.train()
    
    # wandb.finish()


if __name__ == "__main__":
    dataset_list = ['steam', 'nsmc', 'naver']
    base_model_list = ['kykim/bert-kor-base', 'kykim/electra-kor-base', 'kykim/albert-kor-base']
    
    for base_model_name in base_model_list:    
        for dname in dataset_list:
            train_lmkor(
                base_model_name = base_model_name,
                dataset_name = dname,
                batch_size = 16,
                epoch = 5,
                learning_rate = 5e-5,
            )