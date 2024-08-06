# LLM Fine-Tuning

## Overview

This project is designed for fine-tuning language models using the PyTorch Lightning framework, with a focus on advanced training techniques and optimizations. It is specifically tailored for English and Korean summarization tasks.

## Key Components

- **`l_sweep.py`**: The primary script for setting up and executing the fine-tuning process, including hyperparameter sweeps with Weights & Biases (Wandb).

### Dependencies

Dependencies for this project are listed in the `requirements.txt` file.


### Script Overview

The `l_sweep.py` script performs the following tasks:

1. **Imports Required Libraries**: Uses PyTorch Lightning, Ray, Weights & Biases, and Transformers libraries.
2. **Initial Setup**: Configures environment variables and initializes Wandb.
3. **Training Function**: Defines the `l2ray_trainer` function to set up the model, tokenizer, dataset, and Trainer for fine-tuning. This includes:
   - Loading the model and tokenizer via `get_model()`.
   - Preparing the dataset using `get_dataset()`.
   - Configuring the Trainer with callbacks for early stopping, learning rate monitoring, model checkpointing, and logging.
   - Running model training with `Trainer.fit()`.
4. **Ray Wrapping**: Defines the `ray_wrapped_trainer` function for executing the training on Ray clusters.
5. **Hyperparameter Sweeping**: Uses Wandb to perform hyperparameter optimization, with a sweep configuration to explore various hyperparameter combinations.

### Dataset

The script is set up to fine-tune models on English and Korean summarization datasets. Ensure that these datasets are properly prepared and accessible at jonathankang/ENKO-MEDIQA

### Configuration

- **Model**: Specify the model names from `hf_model_list`.
- **Epochs**: Define the number of epochs for training.
- **Learning Rate**: Set the learning rate range.
- **Gradient Accumulation**: Adjust the number of gradient accumulation steps.
- **Gradient Clipping**: Define the gradient clipping value.
- **LoRA Parameters**: Configure the LoRA parameters such as rank, alpha, dropout, and initialization weights.

### Usage

To execute the script, run:

bash sweep


to stop a run:
ps -ef | grep sweep
kill -9 [sweepnumber]

