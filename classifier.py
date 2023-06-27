import os
os.environ["IPEX_TILE_AS_DEVICE"] = "0"

import torch
import intel_extension_for_pytorch as ipex  # import requried to add xpu namespace to torch.

import logging
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"xpu_device count: {ipex.xpu.device_count()}")
logger.info(f"is tile considered as a device: {ipex.xpu.using_tile_as_device()}")


class TextClassifier:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        num_labels: int,
        task_type: str,
        output_dir: str = "./output",
    ):
        """
        Initializes the Text Classifier.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset_name (str): The name of the dataset to be used for training.
            num_labels (int): The number of unique labels in the dataset.
            task_type (str): The type of classification task. Must be either "multi_class" or "multi_label".
            output_dir (str, optional): The directory where the output files will be saved. Defaults to "./output".

        Dataset Structure:
            The input dataset should be a dictionary-like object with the keys 'text' and 'label'. The 'text' field
            should contain the text data as a list of strings. The 'label' field should contain the label data as a list
            of integers for multi-class classification or as a list of list of integers for multi-label classification.
            For example, for multi-class classification, the dataset can look like:
                {
                    'text': ['This is text 1', 'This is text 2', 'This is text 3'],
                    'label': [0, 2, 1]  # where 0, 1, 2 are class labels
                }
            And for multi-label classification, the dataset can look like:
                {
                    'text': ['This is text 1', 'This is text 2', 'This is text 3'],
                    'label': [[0, 1], [1, 2], [0, 2]]  # where 0, 1, 2 are class labels
                }
        """
        assert task_type in {"multi_class", "multi_label"}
        self.device_name = ipex.xpu.get_device_name()
        logger.info(f"Device name: {self.device_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset_name = dataset_name
        self.num_labels = num_labels
        self.task_type = task_type
        self.output_dir = output_dir
        self.mlb = MultiLabelBinarizer()

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

    def compute_multi_label_metrics(self, eval_pred):
        """mutli label metrics if you have multiple labels per text input."""
        logits, labels = eval_pred
        predictions = torch.sigmoid(torch.tensor(logits)).numpy() >= 0.5
        accuracy_metric = load_metric("accuracy")
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1_metric = load_metric("f1")
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="samples"
        )
        return {"accuracy": accuracy, "f1": f1}

    def compute_multi_class_metrics(self, eval_pred):
        """mutli class metrics - if you call unique classes for the dataset."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy_metric = load_metric("accuracy")
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        return {"accuracy": accuracy}

    def encode_labels(self, labels):
        if self.task_type == "multi_label":
            labels = self.mlb.fit_transform([labels["topics"]])[0]
            return {"labels": labels}
        elif self.task_type == "multi_class":
            return {"labels": labels["label"]}

    def load_and_process_data(self):
        dataset = load_dataset(self.dataset_name)
        dataset = dataset.map(self.encode_labels, batched=True)
        dataset = dataset.map(self.encode, batched=True)
        if self.task_type == "multi_label":
            dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels"]
            )
            self.metric_computer = self.compute_multi_label_metrics
        elif self.task_type == "multi_class":
            dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels"]
            )
            self.metric_computer = self.compute_multi_class_metrics
        dataset = dataset["train"].train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        return train_dataset, val_dataset

    def train(self, epochs=10, batch=16, use_bf16=False):
        train_dataset, val_dataset = self.load_and_process_data()
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch * 4,
            bf16=use_bf16,  # if you want bfloat16 to be used instead float32 (default) for finetuning, pass `use_bf16=True`
            warmup_steps=500,
            weight_decay=0.01,
            report_to="none",
            no_cuda=True,  # set cuda=False on intel gpus
            use_xpu=True,  # set use_xpu=True to use intel accelerator
            use_ipex=True,  # use ipex to optimize the model (optional)
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=50,
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.metric_computer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()
        trainer.evaluate()


if __name__ == "__main__":
    tasks = [
        {
            "dataset": "imdb",
            "model": "distilbert-base-uncased",
            "labels": 2,
            "type": "multi_class",
        },
        {
            "dataset": "ag_news",
            "model": "distilbert-base-uncased",
            "labels": 4,
            "type": "multi_class",
        },
    ]
    for task in tasks:
        logger.info(
            f"Training {task['type']} model: {task['model']} on dataset: {task['dataset']}"
        )
        classifier = TextClassifier(
            task["model"], task["dataset"], task["labels"], task["type"]
        )
        start_time = time.time()
        classifier.train()
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time for {task['model']}: {training_time} seconds")
