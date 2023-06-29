import re
import os

import torch
import intel_extension_for_pytorch

import warnings

warnings.filterwarnings("ignore")


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import wandb

MODEL_NAMES = [
    "albert-base-v2",
    "bert-base-uncased",
    "distilbert-base-uncased",
    "distilroberta-base"
]

sweep_config = {
    "name": "my-sweep",
    "method": "bayes",
    "metric": {"name": "eval_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "min": 3e-4,
            "max": 5e-4,
        },
        "num_train_epochs": {"values": [3, 5]},
        "model": {"values": MODEL_NAMES},
        "batch_size": {"values": [128, 256]},
    },
}

sweep_id = wandb.sweep(sweep_config)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def num_classes(self):
        return len(set(self.labels))

    def __len__(self):
        return len(self.labels)



def clean_data(raw_filepath: str, cleaned_filepath: str) -> pd.DataFrame:
    """Clean the raw data for NLP classification tasks."""
    if os.path.exists(cleaned_filepath):
        cleaned_data = pd.read_csv(cleaned_filepath)
        unique_labels = cleaned_data["label"].unique()
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        cleaned_data["label"] = cleaned_data["label"].map(label_to_int)
    else:
        raw_data = pd.read_csv(raw_filepath)
        raw_data["text1"] = raw_data["text1"].str.replace(",", "\,")
        raw_data["text2"] = raw_data["text2"].str.replace(",", "\,")
        raw_data["label_column"] = raw_data[
            "label_column"
        ].fillna(-1)
        cleaned_data = pd.DataFrame()
        cleaned_data["text"] = (
            #"text1: "
            + raw_data["text1"]
            + ". "
            #+ ". text2: "
            + raw_data["tex2"]
            + "."
        )
        cleaned_data["label"] = raw_data["label_column"].astype("int64")
        unique_labels = cleaned_data["label"].unique()
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        cleaned_data["label"] = cleaned_data["label"].map(label_to_int)
        cleaned_data.to_csv(cleaned_filepath, index=False)
    return cleaned_data, label_to_int


def load_and_preprocess_data(df, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(
        df["text"].tolist(), truncation=True, padding=True, max_length=512
    )
    labels = df["label"].tolist()
    return encodings, labels


def train():
    print("xpu device is available: ", torch.xpu.is_available())
    run = wandb.init(project="nlp_classify_sweep")
    train_encodings, train_labels = load_and_preprocess_data(train_df, run.config.model)
    val_encodings, val_labels = load_and_preprocess_data(test_df, run.config.model)
    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)
    num_classes = train_dataset.num_classes()
    print(f'Number of unique classes: {num_classes}')
    model = AutoModelForSequenceClassification.from_pretrained(
        run.config.model, num_labels=num_classes
    )
    training_args = TrainingArguments(
        output_dir=f"./results/{wandb.run.id}",
        num_train_epochs=run.config.num_train_epochs,
        per_device_train_batch_size=run.config.batch_size,
        per_device_eval_batch_size=run.config.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs{wandb.run.id}",
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        ddp_find_unused_parameters=None,
        optim="adamw_torch",
        logging_steps=25,
        learning_rate=run.config.learning_rate,
        gradient_accumulation_steps=4,
        no_cuda=True,
        use_xpu=True,
        use_ipex=True,
    )
    print("starting training.. ")
    trainer = Trainer(
        model=model.to(device=torch.device("xpu"), dtype=torch.float32),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    eval_results = trainer.evaluate()
    wandb.log({"eval_loss": eval_results["eval_loss"]})
    trainer.save_model(f"./models/{wandb.run.id}")
    run.finish()


if __name__ == "__main__":
    train_df, labels_map_train = clean_data(
        "raw_train.csv", "cleaned_train_data.csv"
    )
    test_df, labels_map_test = clean_data(
        "raw_valid.csv", "cleaned_test_data.csv"
    )
    wandb.agent(sweep_id, train)
