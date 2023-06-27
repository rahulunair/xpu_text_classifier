import requests
import random
import numpy as np
from datasets import Dataset
import os

DEFAULT_BOOK_URL = "http://www.gutenberg.org/files/1342/1342-0.txt"


def load_gutenberg_book(book_url=DEFAULT_BOOK_URL):
    """load a default book from gutenberg."""
    response = requests.get(book_url)
    text = response.content.decode(errors="ignore")
    return text


def create_random_text_samples(text, n_samples, max_length):
    """split the book into random text samples to act as input."""
    words = text.split()
    text_samples = []
    for _ in range(n_samples):
        start_index = random.randint(0, len(words) - max_length)
        sample = " ".join(words[start_index : start_index + max_length])
        text_samples.append(sample)
    return text_samples


def prepare_gutenberg_dataset(
    n_samples=1000, n_classes=30, max_length=200, book_url=DEFAULT_BOOK_URL
):
    """prepare a dataset with the random text samples as input and also random labels between 0 and 'n_samples.'."""
    text = load_gutenberg_book(book_url)
    text_samples = create_random_text_samples(text, n_samples, max_length)
    labels = np.random.randint(0, n_classes, n_samples).tolist()
    dataset = Dataset.from_dict({"text": text_samples, "label": labels})
    return dataset


def save_dataset_to_disk(dataset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset.to_csv(os.path.join(output_dir, "dataset.csv"))
