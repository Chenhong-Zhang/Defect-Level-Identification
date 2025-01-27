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
        self.mode = mode  # 取值为 'train'、'val' 或 'test'

        # Mapping feature names from Chinese to English
        self.feature_name_mapping = {
            '病害描述': 'defect_description',
            '部件': 'component',
            '预测病害数量修正': 'defect_number_correction',
            '裂缝宽度': 'crack_width',
            '病害面积': 'defect_area',
            '病害长度': 'defect_length',
            '预测病害数量指示': 'defect_number_indicator',
            '其他指示符': 'dimension_indicator'
        }

        # Text feature names
        self.text_features = [
            '病害描述',
            '部件',
        ]

        # Group for "defect_dimension_numerical"
        self.dimension_numerical_features = [
            '裂缝宽度',           # Crack Width
            '病害面积',           # Defect Area
            '病害长度',           # Defect Length
            '其他指示符'          # Dimension Indicator
        ]

        # Group for "defect_number"
        self.defect_number_features = [
            '预测病害数量修正',    # Defect Number Correction
            '预测病害数量指示'     # Defect Number Indicator
        ]

        # Labels
        self.labels = self.data['病害标度'].values

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
            english_feature_name = self.feature_name_mapping[feature_name]
            # Flatten tensors and use English keys
            encoded_inputs[f'input_ids_{english_feature_name}'] = encoding['input_ids'].squeeze(0)
            encoded_inputs[f'attention_mask_{english_feature_name}'] = encoding['attention_mask'].squeeze(0)

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
        self.mode = mode  # 取值为 'train'、'val' 或 'test'

        # Mapping feature names from Chinese to English
        self.feature_name_mapping = {
            '病害描述': 'defect_description',
            '部件': 'component',
            '预测病害数量修正': 'defect_number_correction',
            '裂缝宽度': 'crack_width',
            '病害面积': 'defect_area',
            '病害长度': 'defect_length',
            '预测病害数量指示': 'defect_number_indicator',
            '其他指示符': 'dimension_indicator'
        }

        # Text feature names
        self.text_features = [
            '病害描述',
            '部件',
        ]

        # Group for "defect_dimension_numerical"
        self.dimension_numerical_features = [
            '裂缝宽度',           # Crack Width
            '病害面积',           # Defect Area
            '病害长度',           # Defect Length
            '其他指示符'          # Dimension Indicator
        ]

        # Group for "defect_number"
        self.defect_number_features = [
            '预测病害数量修正',    # Defect Number Correction
            '预测病害数量指示'     # Defect Number Indicator
        ]

        # Labels
        self.labels = self.data['病害标度'].values 
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
            english_feature_name = self.feature_name_mapping[feature_name]
            # Flatten tensors and use English keys
            encoded_inputs[f'input_ids_{english_feature_name}'] = encoding['input_ids'].squeeze(0)
            encoded_inputs[f'attention_mask_{english_feature_name}'] = encoding['attention_mask'].squeeze(0)

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
        self.mode = mode  # 取值为 'train'、'val' 或 'test'

        # Mapping feature names from Chinese to English
        self.feature_name_mapping = {
            '病害描述': 'defect_description',
            '部件': 'component',
            '预测病害数量修正': 'defect_number_correction',
            '裂缝宽度': 'crack_width',
            '病害面积': 'defect_area',
            '病害长度': 'defect_length',
            '预测病害数量指示': 'defect_number_indicator',
            '其他指示符': 'dimension_indicator'
        }

        # Text feature names
        self.text_features = [
            '病害描述',
            '部件',
        ]

        # Group for "defect_dimension_numerical"
        self.dimension_numerical_features = [
            '裂缝宽度',           # Crack Width
            '病害面积',           # Defect Area
            '病害长度',           # Defect Length
            '其他指示符'          # Dimension Indicator
        ]

        # Group for "defect_number"
        self.defect_number_features = [
            '预测病害数量修正',    # Defect Number Correction
            '预测病害数量指示'     # Defect Number Indicator
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
            english_feature_name = self.feature_name_mapping[feature_name]
            # Flatten tensors and use English keys
            encoded_inputs[f'input_ids_{english_feature_name}'] = encoding['input_ids'].squeeze(0)
            encoded_inputs[f'attention_mask_{english_feature_name}'] = encoding['attention_mask'].squeeze(0)

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
