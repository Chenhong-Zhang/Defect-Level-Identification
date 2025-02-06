import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, scaler_save_path, mode='train'):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode  # 'train'、'val' 或 'test'

        self.feature_name_mapping = {
            'Defect Description': 'defect_description',
            'Component': 'component',            
        }

        
        # Text feature names
        self.text_features = [
            'Defect Description',
            'Component',
        ]

        # Group for "defect_dimension_numerical"
        self.dimension_numerical_features = [
            'Defect Width', 
            'Defect Area',
            'Defect Length',
            'Dimension Specified' 
        ]

        # Group for "defect_number"
        self.defect_number_features = [
            'Defect Quantity',
            'Quantity Specified'
        ]

        # Labels
        self.labels = self.data['Defect Level'].values

        # Initialize scalers
        self.scalers = {}

        # For numerical features, we need to fit scalers on training data and load scalers on validation/test data
        numerical_features = self.dimension_numerical_features + self.defect_number_features

        for feature_name in numerical_features:
            scaler_filename = os.path.join(scaler_save_path, f'{feature_name}_scaler.pkl')

            if self.mode == 'train':
                # Fit the scaler and save it
                scaler = StandardScaler()
                # Reshape the data for the scaler as it expects 2D input
                feature_data = self.data[feature_name].values.reshape(-1, 1)
                scaler.fit(feature_data)
                self.scalers[feature_name] = scaler
                # Save the scaler for later use
                joblib.dump(scaler, scaler_filename)
            else:
                # Load the scaler
                scaler = joblib.load(scaler_filename)
                self.scalers[feature_name] = scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # Tokenize text features and map keys to English
        encoded_inputs = {}
        for feature_name in self.text_features:
            text = str(sample[feature_name])
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            # Map Chinese feature name to English
            feature_name = self.feature_name_mapping[feature_name]
            # Flatten tensors and use English keys
            encoded_inputs[f'input_ids_{feature_name}'] = encoding['input_ids'].squeeze(0)
            encoded_inputs[f'attention_mask_{feature_name}'] = encoding['attention_mask'].squeeze(0)

        # Prepare "defect_dimension_numerical" group
        defect_dimension_numerical = []
        for feature_name in self.dimension_numerical_features:
            value = sample[feature_name]
            # Reshape to (1, 1) for scaler
            value_array = np.array([[value]])
            scaled_value = self.scalers[feature_name].transform(value_array)[0][0]
            defect_dimension_numerical.append(scaled_value)
        defect_dimension_numerical = torch.tensor(defect_dimension_numerical, dtype=torch.float)

        # Prepare "defect_number" group
        defect_number = []
        for feature_name in self.defect_number_features:
            value = sample[feature_name]
            # Reshape to (1, 1) for scaler
            value_array = np.array([[value]])
            scaled_value = self.scalers[feature_name].transform(value_array)[0][0]
            defect_number.append(scaled_value)
        defect_number = torch.tensor(defect_number, dtype=torch.float)

        # Prepare label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Combine all features into a single dictionary
        item = {
            **encoded_inputs,
            'defect_dimension_numerical': defect_dimension_numerical,
            'defect_number': defect_number,
            'labels': label,
            'idx': idx
        }

        return item



class TestDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, scaler_save_path, mode='train'):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode  # 'train'、'val' 或 'test'

        self.feature_name_mapping = {
            'Defect Description': 'defect_description',
            'Component': 'component',            
        }     

        # Text feature names
        self.text_features = [
            'Defect Description',
            'Component',
        ]

        # Group for "defect_dimension_numerical"
        self.dimension_numerical_features = [
            'Defect Width', 
            'Defect Area',
            'Defect Length',
            'Dimension Specified' 
        ]

        # Group for "defect_number"
        self.defect_number_features = [
            'Defect Quantity',
            'Quantity Specified'
        ]

        # Labels
        self.labels = self.data['Defect Level'].values 
        self.ambiguous_labels = self.data['Ambiguous'].values

        # Initialize scalers
        self.scalers = {}

        # For numerical features, we need to fit scalers on training data and load scalers on validation/test data
        numerical_features = self.dimension_numerical_features + self.defect_number_features

        for feature_name in numerical_features:
            scaler_filename = os.path.join(scaler_save_path, f'{feature_name}_scaler.pkl')

            if self.mode == 'train':
                # Fit the scaler and save it
                scaler = StandardScaler()
                # Reshape the data for the scaler as it expects 2D input
                feature_data = self.data[feature_name].values.reshape(-1, 1)
                scaler.fit(feature_data)
                self.scalers[feature_name] = scaler
                # Save the scaler for later use
                joblib.dump(scaler, scaler_filename)
            else:
                # Load the scaler
                scaler = joblib.load(scaler_filename)
                self.scalers[feature_name] = scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # Tokenize text features and map keys to English
        encoded_inputs = {}
        for feature_name in self.text_features:
            text = str(sample[feature_name])
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            # Map Chinese feature name to English
            feature_name = self.feature_name_mapping[feature_name]
            # Flatten tensors and use English keys
            encoded_inputs[f'input_ids_{feature_name}'] = encoding['input_ids'].squeeze(0)
            encoded_inputs[f'attention_mask_{feature_name}'] = encoding['attention_mask'].squeeze(0)

        # Prepare "defect_dimension_numerical" group
        defect_dimension_numerical = []
        for feature_name in self.dimension_numerical_features:
            value = sample[feature_name]
            # Reshape to (1, 1) for scaler
            value_array = np.array([[value]])
            scaled_value = self.scalers[feature_name].transform(value_array)[0][0]
            defect_dimension_numerical.append(scaled_value)
        defect_dimension_numerical = torch.tensor(defect_dimension_numerical, dtype=torch.float)

        # Prepare "defect_number" group
        defect_number = []
        for feature_name in self.defect_number_features:
            value = sample[feature_name]
            # Reshape to (1, 1) for scaler
            value_array = np.array([[value]])
            scaled_value = self.scalers[feature_name].transform(value_array)[0][0]
            defect_number.append(scaled_value)
        defect_number = torch.tensor(defect_number, dtype=torch.float)

        # Prepare label
        label = torch.tensor(
            [self.labels[idx], self.ambiguous_labels[idx]], dtype=torch.long
        )
        
        # Combine all features into a single dictionary
        item = {
            **encoded_inputs,
            'defect_dimension_numerical': defect_dimension_numerical,
            'defect_number': defect_number,
            'labels': label,            
            'idx': idx
        }

        return item
    

class PredictDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, scaler_save_path, mode='test'):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode  # 'train'、'val' 或 'test'

        self.feature_name_mapping = {
            'Defect Description': 'defect_description',
            'Component': 'component',            
        }

        # Text feature names
        self.text_features = [
            'Defect Description',
            'Component',
        ]

        # Group for "defect_dimension_numerical"
        self.dimension_numerical_features = [
            'Defect Width', 
            'Defect Area',
            'Defect Length',
            'Dimension Specified' 
        ]

        # Group for "defect_number"
        self.defect_number_features = [
            'Defect Quantity',
            'Quantity Specified'
        ]

        # Initialize scalers
        self.scalers = {}

        # For numerical features, we need to fit scalers on training data and load scalers on validation/test data
        numerical_features = self.dimension_numerical_features + self.defect_number_features

        for feature_name in numerical_features:
            scaler_filename = os.path.join(scaler_save_path, f'{feature_name}_scaler.pkl')

            if self.mode == 'train':
                # Fit the scaler and save it
                scaler = StandardScaler()
                # Reshape the data for the scaler as it expects 2D input
                feature_data = self.data[feature_name].values.reshape(-1, 1)
                scaler.fit(feature_data)
                self.scalers[feature_name] = scaler
                # Save the scaler for later use
                joblib.dump(scaler, scaler_filename)
            else:
                # Load the scaler
                scaler = joblib.load(scaler_filename)
                self.scalers[feature_name] = scaler
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # Tokenize text features and map keys to English
        encoded_inputs = {}
        for feature_name in self.text_features:
            text = str(sample[feature_name])
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            # Map Chinese feature name to English
            feature_name = self.feature_name_mapping[feature_name]
            # Flatten tensors and use English keys
            encoded_inputs[f'input_ids_{feature_name}'] = encoding['input_ids'].squeeze(0)
            encoded_inputs[f'attention_mask_{feature_name}'] = encoding['attention_mask'].squeeze(0)

        # Prepare "defect_dimension_numerical" group
        defect_dimension_numerical = []
        for feature_name in self.dimension_numerical_features:
            value = sample[feature_name]
            # Reshape to (1, 1) for scaler
            value_array = np.array([[value]])
            scaled_value = self.scalers[feature_name].transform(value_array)[0][0]
            defect_dimension_numerical.append(scaled_value)
        defect_dimension_numerical = torch.tensor(defect_dimension_numerical, dtype=torch.float)

        # Prepare "defect_number" group
        defect_number = []
        for feature_name in self.defect_number_features:
            value = sample[feature_name]
            # Reshape to (1, 1) for scaler
            value_array = np.array([[value]])
            scaled_value = self.scalers[feature_name].transform(value_array)[0][0]
            defect_number.append(scaled_value)
        defect_number = torch.tensor(defect_number, dtype=torch.float)
        
        # Combine all features into a single dictionary
        item = {
            **encoded_inputs,
            'defect_dimension_numerical': defect_dimension_numerical,
            'defect_number': defect_number,            
            'idx': idx
        }

        return item
