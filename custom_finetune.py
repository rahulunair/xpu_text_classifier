import torch
import intel_extension_for_pytorch

import os

os.environ["IPEX_TILE_AS_DEVICE"] = "0"

import data_utils
from classifier import TextClassifier

model_name = "distilbert-base-uncased"  # use any transformer model like: "facebook/bart-large", "bert-base-uncased", "distilbert-base-uncased", "albert-base-v2", "distilbart-base", "roberta-base", "xlnet-base-cased","distilroberta-base"
dataset_name = (
    "gutenberg_dataset"  # directory with csv file with 2 columns: text, label
)
num_labels = 30
task_type = "multi_class"
output_dir = "./output"
epochs = 30
batch_size=128

"""
Here we have prepared a dataset with input being an ebook from guttenberg,and,
the classes are integers between 0 and 29 (transformers expect the classes to be
integers). Once the file is downloaded its stored in the dir "gutenberg_dataset"
as a csv. If your labels are strings, then using a label encoder to conver them
to integers before passing it to the TextClassifier:

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = ['cat', 'mat', 'bat', 'cat', 'bat']
    encoded_labels = le.fit_transform(labels)
"""

if __name__ == "__main__":
    print("_______")
    print("fetching and preparing a book from gutenberg.")
    # fetch and prep dataset
    dataset = data_utils.prepare_gutenberg_dataset()
    print(f"saving dataset to directory: {dataset_name}")
    data_utils.save_dataset_to_disk(dataset, dataset_name)
    print("use custom data to train a text classifier")
    print(f"using model: {model_name}")
    print(f"task type: {task_type}")
    print(f"number of labels in the dataset: {num_labels}")
    print("--------")
    classifier = TextClassifier(
        model_name, dataset_name, num_labels, task_type, output_dir
    )
    classifier.train(epochs=epochs, batch=batch_size)
