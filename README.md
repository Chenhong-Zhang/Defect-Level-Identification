# Defect-Level-Identification

This repository contains the implementation of the defect level identification model described in the paper:  
**_A Machine Learning Approach for Enhancing Bridge Inspection Data Quality_**.

## Overview

The purpose of this repository is to demonstrate the organization and execution of the defect level identification model. Due to restrictions on the research data used in the paper, the dataset provided here is for demonstration purposes only. It illustrates how to structure your training, testing, and validation data for similar tasks.

## Repository Contents

- **`CustomDataset.py`**: Script for handling and preprocessing dataset inputs.
- **`Model.py`**: The implementation of the defect level identification model.
- **`Tools.py`**: Utility functions and tools.
- **`main.py`**: The main script to execute the training and testing.
- **`train_single.py`**: Script for training the a single base model. 
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
git clone https://github.com/Chenhong-Zhang/Defect-Level-Identification.git
cd Defect-Level-Identification
```

Then run the code:

```bash
python main.py --config configs.json
```

## Configuration Details

This section provides a detailed explanation of the configurations specified in the `configs.json` file.


### **General Configurations**

Adjust there parameters according to your training settings.

- **`batch_size`**  
  The number of samples processed in one forward/backward pass during training. Adjust it based on your GPU RAM.  

- **`num_epochs`**  
  The total number of training cycles the model will go through.    

- **`learning_rate`**  
  The step size at which the model updates its parameters during training.
  
- **`dropout_rate`**  
  Dropout rate applied to prevent overfitting. 

- **`early_stopping_patience`**  
  Number of epochs without improvement in validation performance before training stops early.   

- **`num_classes`**  
  Number of output classes for the classification task.

- **`max_grad_norm`**  
  A gradient clipping parameter to prevent exploding gradients during training. Gradients are scaled to have a maximum norm of `1.0`.   


### **Paths**

Adjust there parameters according to directories.

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


### **Sequence and Model Settings**

Do not adjust there parameters unless you know what you are doing.

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


### **Parameters For the Customized Training Strategy**

Try optimize these parameters based on experiments on your own dataset.

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

## Data Organization

### Datasets Overview

There are two datasets provided for demonstration:

- **Training Data**: Contains 9 fields.
- **Test/Validation Data**: Contains 10 fields, with an additional *Ambiguous* column.

### Feature Descriptions

Below is a detailed explanation of each input feature in the dataset:

| Feature | Explanation | Data Type | Example Value |
|---------|------------|-----------|---------------|
| Defect Description | Text description of the defects. | string | At the edge of beam 17# and beam 18#, there are 3 instances of exposed reinforcement at the wet joints, each covering a 0.1m x 0.2m area. |
| Component | The component where the defect is located. | string | Load-Bearing Component |
| Quantity Specified | Indicates if the defect quantity is explicitly provided. | boolean | Yes (represented as 1) |
| Defect Quantity | Specified quantity of the defects. | integer | 3 |
| Dimension Specified | Indicates if the defect dimension is explicitly provided. | boolean | Yes (represented as 1) |
| Defect Width | Width of the defect (for crack-related defects only). Unit: mm. | float | 0.01 (i.e., 0.01mm) |
| Defect Length | Length of the defect. Unit: mm. | integer | 100 (i.e., 100mm) |
| Defect Area | Area of the defect. Unit: mm². | integer | 50000 (i.e., 50000mm²) |
| Defect Level | Identified Level. | integer | 2 |
| Ambiguous | Whether the defect is ambiguous | bool | Yes (represented as 1) |




