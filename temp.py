from main import load_data

import evaluate
import datasets
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

dataset = load_data(train_file_path = './data/nsmc_train.csv', test_file_path = './data/nsmc_test.csv', n_train = 30_000)

tokenizer = AutoTokenizer.from_pretrained('monologg/kobert')
model = AutoModelForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2).to('cuda')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

batchsize = 16
epoch = 5
learning_rate = 5e-5

training_args = TrainingArguments(
    output_dir = './results/kobert-naver',
    per_device_train_batch_size = batchsize,
    per_device_eval_batch_size = batchsize,
    learning_rate = learning_rate,
    num_train_epochs = epoch,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    logging_steps = 50,
)

metric = evaluate.combine(['accuracy', 'f1', 'recall', 'precision'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

