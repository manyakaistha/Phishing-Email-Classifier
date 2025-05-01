import re
import torch
import torch.nn as nn
import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s\.\?,!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def combine_features(example):
    email_text = clean_text(example.get('Email Text', ''))
    example['text'] = email_text
    return example

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

class DistilBertLSTMClassifier(nn.Module):
    def __init__(self, distilbert_model_name='distilbert-base-uncased', lstm_hidden_size=128, lstm_layers=1, dropout_rate=0.2, num_labels=1):
        super().__init__()
        self.num_labels = num_labels
        self.distilbert = DistilBertModel.from_pretrained(distilbert_model_name)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(input_size=self.distilbert.config.hidden_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout_rate if lstm_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = distilbert_output.last_hidden_state

        lstm_output, (hidden, cell) = self.lstm(sequence_output)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_output.size()).float()
        sum_embeddings = torch.sum(lstm_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        final_hidden_state = pooled_output

        dropped_output = self.dropout(final_hidden_state)

        logits = self.classifier(dropped_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(), labels.float().squeeze())

        if loss is not None:
             return {"loss": loss, "logits": logits}
        else:
             return {"logits": logits}

def compute_metrics(pred):
    labels = pred.label_ids
    preds_probs = torch.sigmoid(torch.from_numpy(pred.predictions)).numpy()
    preds = (preds_probs > 0.5).astype(int).flatten()

    labels = labels.flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

if __name__ == "__main__":
    dataset_id = 'zefang-liu/phishing-email-dataset'
    print(f"Loading dataset: {dataset_id}")
    ds = load_dataset(dataset_id)
    print("Dataset loaded:")
    print(ds)

    print("Cleaning and combining text features...")
    ds = ds.map(combine_features, batched=False)
    print("Dataset after cleaning and combining:")
    print(ds['train'][0])

    tokenizer_id = 'distilbert-base-uncased'
    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_id)

    print("Tokenizing dataset...")
    tokenized_ds = ds.map(tokenize_function, batched=True)

    def map_label(example):
        example['labels'] = 1 if example['Email Type'] == 'Phishing Email' else 0
        return example
    print("Mapping labels to integers...")
    tokenized_ds = tokenized_ds.map(map_label, batched=False)

    columns_to_remove = ['Unnamed: 0', 'Email Text', 'Email Type', 'text']
    tokenized_ds_before_format = tokenized_ds.remove_columns(columns_to_remove)

    tokenized_ds_before_format.set_format("torch")

    print("Tokenized dataset ready:")
    print(tokenized_ds_before_format['train'][0])

    tokenized_ds = tokenized_ds_before_format

    print("Instantiating DistilBertLSTMClassifier model...")
    model = DistilBertLSTMClassifier(num_labels=1)
    print("Model instantiated.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"MPS device found. Moving model to {device}")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")

    output_dir = "./phishing_results"
    logging_dir = './logs'

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=6,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        warmup_steps=500,
        logging_dir=logging_dir,
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        dataloader_num_workers=2,
        report_to="none",
    )
    print("TrainingArguments configured.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("Data collator initialized.")

    train_dataset = tokenized_ds['train']
    if 'test' in tokenized_ds:
        eval_dataset = tokenized_ds['test']
        print("Using 'test' split for evaluation.")
    else:
        print("No 'test' split found. Splitting 'train' set for evaluation (80/20).")
        split_ds = tokenized_ds['train'].train_test_split(test_size=0.2, seed=42)
        train_dataset = split_ds['train']
        eval_dataset = split_ds['test']

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    print("Trainer initialized. Ready for training.")

    print("Starting training...")
    train_result = trainer.train()

    print("Training finished.")
    trainer.save_model(output_dir + "/final_model")
    tokenizer.save_pretrained(output_dir + "/final_model")
    print(f"Final model saved to {output_dir}/final_model")

    print("Evaluating the best model on the test set...")
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    print("Evaluation results:")
    print(eval_results)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Training finished successfully.")