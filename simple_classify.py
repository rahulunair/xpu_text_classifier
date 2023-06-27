import os
os.environ["IPEX_TILE_AS_DEVICE"] = "0"

import logging
import time
import warnings

warnings.filterwarnings("ignore")

import intel_extension_for_pytorch as ipex
import torch


from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"xpu_device count: {ipex.xpu.device_count()}")
logger.info(f"is tile considered as a device: {ipex.xpu.using_tile_as_device()}")

class TextClassifier:
    def __init__(
        self, model_name: str, dataset_name: str, output_dir: str = "./output"
    ):
        self.device_name = ipex.xpu.get_device_name()
        logger.info(f"Device name: {self.device_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset_name = dataset_name
        self.output_dir = output_dir

    def encode(self, examples):
        tokenized_example = self.tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )
        if "token_type_ids" in tokenized_example:
            return {
                "input_ids": tokenized_example["input_ids"],
                "attention_mask": tokenized_example["attention_mask"],
                "token_type_ids": tokenized_example["token_type_ids"],
            }
        else:
            return {
                "input_ids": tokenized_example["input_ids"],
                "attention_mask": tokenized_example["attention_mask"],
            }

    def train(self):
        dataset = load_dataset(self.dataset_name)
        dataset = dataset.map(self.encode, batched=True)
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        dataset = dataset["train"].train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            report_to="none",
            no_cuda=True,
            use_xpu=True,
            use_ipex=True,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=500
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Added
        )
        trainer.train()
        trainer.evaluate()


if __name__ == "__main__":
    model_names = [
        "facebook/bart-large",
        "bert-base-uncased",
        "distilbert-base-uncased",
        "distilbart-base",
        "roberta-base",
        "xlnet-base-cased",
        "albert-base-v2",
        "distilroberta-base",
    ]
    dataset_name = "imdb"
    for model_name in model_names:
        logger.info(f"Training model: {model_name}")
        classifier = TextClassifier(model_name, dataset_name)
        start_time = time.time()
        classifier.train()
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time for {model_name}: {training_time} seconds")
