# Defect-Level-Identification

This repository contains the implementation of the defect level identification model described in the paper:  
**_A Machine Learning Approach for Enhancing Bridge Inspection Data Quality_**.

## Overview

The purpose of this repository is to demonstrate the organization and execution of the defect level identification model. Due to restrictions on the research data used in the paper, the dataset provided here is for demonstration purposes only. It illustrates how to structure your training, testing, and validation data for similar tasks.

## Repository Contents

- **`CustomDataset.py`**: Script for handling and preprocessing dataset inputs.
- **`Model.py`**: Contains the implementation of the defect level identification model.
- **`Tools.py`**: Includes utility functions and tools for data processing and model evaluation.
- **`main.py`**: The main script to execute the model.
- **`train_single.py`**: Script for training the model on a single dataset or configuration.
- **`configs.json`**: Configuration file for specifying model parameters, data paths, and other settings.
- **`requirements.txt`**: Lists the dependencies required for the project.
- **`README.md`**: Documentation of the repository.
- **`data/`**: Directory for storing synthetic training, testing, and validation datasets.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.10 or higher
- Required libraries (listed in `requirements.txt`)

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Runing The Model:

First Clone the Repository:

```bash
git clone https://github.com/your-username/Defect-Level-Identification.git
cd Defect-Level-Identification
```

Then run the code:

```bash
python main.py --config configs.json
```

## Configuration Details

This section provides a detailed explanation of the configurations specified in the `configs.json` file.

---

### **General Configurations**
- **`batch_size`**  
  The number of samples processed in one forward/backward pass during training. Adjust it based on your GPU RAM.  

- **`num_epochs`**  
  The total number of training cycles the model will go through.    

- **`learning_rate`**  
  The step size at which the model updates its parameters during training.   

- **`early_stopping_patience`**  
  Number of epochs without improvement in validation performance before training stops early.   

- **`num_classes`**  
  Number of output classes for the classification task.    

---

### **Optimization Configurations**
- **`max_grad_norm`**  
  A gradient clipping parameter to prevent exploding gradients during training. Gradients are scaled to have a maximum norm of `1.0`.   

---

### **Paths**
- **`model_save_path`**  
  Directory path where the trained models will be saved.    

- **`bert_path`**  
  Path to the pre-trained BERT model used for feature extraction.    

- **`data_path`**  
  Path to the directory containing training, validation, and testing datasets.    

- **`val_data_path`**  
  File path to the validation dataset.    

- **`test_data_path`**  
  File path to the testing dataset.   

- **`train_file_path`**  
  Path to the training script used to run the model.    

---

## **Sequence and Model Settings**
- **`max_seq_len`**  
  Maximum length of input sequences for the model. Longer sequences will be truncated.    

- **`projected_dim`**  
  Dimension to which input embeddings are projected before further processing.   

- **`num_layers`**  
  Number of layers in the model architecture.    

- **`growth_rate`**  
  Growth rate for dense layers in the architecture.    

- **`transition_layers`**  
  List specifying the dimensions for the transition layers in the model.    

---

## **Regularization Configurations**
- **`dropout_rate`**  
  Dropout rate applied to prevent overfitting. 

---

## **Parameters For the Customized Training Strategy**
- **`am_percentile`**  
  Percentile value used in AM threshold calculations.  

- **`max_smoothing`**  
  Maximum smoothing parameter for loss smoothing.  

- **`smoothing_scale`**  
  Scale factor for loss smoothing.   

- **`hold_epoch`**  
  Number of epochs before activating the dynamic partitioning.  

- **`num_base_learners`**  
  Number of base learners used for ensemble.  

---


