# xpu_text_classifier: Custom Text Classification on Intel dGPUs

xpu_text_classifier allows you to fine-tune transformer models using custom datasets for multi-class or multi-label classification tasks. The models supported include popular transformer architectures like BERT, BART, DistilBERT, etc. This solution uses the Huggingface Trainer to handle the training and leverages Intel Extension for PyTorch to run on Intel dGPUs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Monitoring GPU Usage](#monitoring-gpu-usage)
- [Additional Details](#additional-details)

## Installation

Before you start, ensure you have PyTorch and Intel Extension for PyTorch installed. 

To install xpu_text_classifier:

1. Clone the transformers_xpu repository from GitHub:

    ```bash
    git clone https://github.com/rahulunair/transformers_xpu.git
    cd transformers_xpu
    ```

2. Install the package:

    ```bash
    python setup.py install
    ```

3. Install the required dependencies:

    ```bash
    pip install datasets scikit-learn
    ```

4. Optionally, install Weights & Biases to monitor your training process:

    ```bash
    pip install wandb
    ```

## Preparing Your Dataset

The dataset should be in a format compatible with the Hugging Face's load_dataset function, which includes CSV, JSON, and several others. The dataset should have two columns 'text' and 'label'.
For multi-class classification tasks, each label is a single integer. For multi-label classification tasks, each label is a list of integers.

Multi-Class Classification Example:


| text            | label |
|-----------------|-------|
| This is text 1  | 0     |
| This is text 2  | 2     |
| This is text 3  | 1     |

Multi-Label Classification Example:

| text            | label        |
|-----------------|--------------|
| This is text 1  | [0, 1]       |
| This is text 2  | [1, 2]       |
| This is text 3  | [0, 2]       |


## Usage

The script `custom_finetune.py` in the root directory is your entry point for training a model. By default, it uses the 'distilbert-base-uncased' model and Gutenberg dataset with 30 labels.

To run the training on a single GPU:

```bash
python custom_finetune.py
```
To run the training using all available GPUs:

```bash
export MASTER_ADDR=127.0.0.1
source /home/orange/pytorch_xpu_orange/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/env/setvars.sh
mpirun -n 4 python custom_finetune.py
```
Replace 4 with the number of GPUs available in your system.

You can specify the model name, number of labels(classes), number of epochs, batch size, and whether to use BF16 precision with the train function. 

## Monitoring GPU Usage

To monitor the GPU usage:

```bash
xpu-smi dump -m5,18  # VRAM utilization
```
## Additional Details

The custom_finetune.py script fetches an e-book from Gutenberg and prepares a dataset for the training task. The dataset is stored in the directory specified by dataset_name as a csv file with two columns: text and label.
**Please note**, the transformers expect the labels to be integers. If your labels are strings, make sure to encode them into integers before passing them to the TextClassifier:

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = ['cat', 'mat', 'bat', 'cat', 'bat']
encoded_labels = le.fit_transform(labels)
```
For more details on the TextClassifier, refer to classifier.py.

Remember to check the script and adjust the parameters (model type, dataset, epochs, batch size, etc.) according to your needs.

Happy fine-tuning!
