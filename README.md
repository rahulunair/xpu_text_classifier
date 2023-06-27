### xpu_text_classifier

Use transformer models to finetune multi-class or mult-label custom datasets.

#### Setup
Once pytorch and intel extension for pytorch has been installed, go to [transformers_xpu](https://github.com/rahulunair/transformers_xpu.git) and install it.

```bash
git clone https://github.com/rahulunair/transformers_xpu.git
cd transformers_xpu
python setup.py install
pip install datasets
pip install scikit-learn
pip install wandb  # optional
```

#### Running

To run on a single GPU:

```bash
python custom_finetune.py
```

To run using all available GPUs:

```bash
export MASTER_ADDR=127.0.0.1
source /home/orange/pytorch_xpu_orange/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/env/setvars.sh
mpirun -n 4 python custom_finetune.py
```

Please monitor the GPU usage using `xpu-smi`:

```bash
xpu-smi dump -m5,18  # VRAM utilization
```

