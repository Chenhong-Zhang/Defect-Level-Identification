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
