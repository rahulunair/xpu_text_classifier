import torch
import intel_extension_for_pytorch

import warnings
warnings.filterwarnings("ignore")


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import wandb

def load_and_preprocess_data(df, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=512)
    return train_encodings, val_encodings

MODEL_NAMES = ["distilbert-base-uncased", "bert-base-uncased", "distilroberta-base", "google/mobilebert-uncased", "albert-base-v2", "squeezebert/squeezebert-uncased", "prajjwal1/bert-mini"]
LEARNING_RATES = [5e-5, 3e-5, 1e-5]

df = pd.read_csv("sample_dataset.csv")
labels_map = {label: i for i, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(labels_map)

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

wandb.login()


# Wandb sweep configuration
sweep_config = {
    'name': 'my-sweep', 
    'method': 'bayes', 
    'metric': {
      'name': 'eval_loss',
      'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 1e-5,
            'max': 5e-5
        },
        'num_train_epochs': {
            'values': [200, 300, 400, 600, 800, 1000]
        },
        'model': {
            'values': MODEL_NAMES
        }
    }
}

sweep_id = wandb.sweep(sweep_config)

def train(df):
    run = wandb.init()
    train_encodings, val_encodings = load_and_preprocess_data(df, run.config.model)
    train_dataset = Dataset(train_encodings, train_labels.tolist())
    val_dataset = Dataset(val_encodings, val_labels.tolist())
    model = AutoModelForSequenceClassification.from_pretrained(run.config.model, num_labels=len(df["label"].unique()))
    training_args = TrainingArguments(
        output_dir=f"./results/{wandb.run.id}",
        num_train_epochs=run.config.num_train_epochs,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    eval_results = trainer.evaluate()
    wandb.log({"eval_loss": eval_results['eval_loss']})
    trainer.save_model(f'./models/{wandb.run.id}')
    run.finish()

if __name__ == "__main__:
  wandb.agent(sweep_id, lambda: train(df))
