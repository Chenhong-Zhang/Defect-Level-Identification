import torch
import torch.nn as nn
from transformers import BertModel, PreTrainedModel
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention_linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        attn_scores = self.attention_linear(hidden_states)  # (batch_size, seq_len, 1)
        attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        # Use clamp to avoid extremely small negative values
        attention_mask = torch.clamp(attention_mask, min=0)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, torch.finfo(attn_scores.dtype).min)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)
        pooled_output = torch.sum(attn_weights * hidden_states, dim=1)  # (batch_size, hidden_size)
        return pooled_output

class DenseLayer(nn.Module):
    def __init__(self, input_dim, growth_rate, dropout_rate=0.2):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.bn(x)
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = torch.cat([x, out], dim=1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_dim, growth_rate, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.output_dim = input_dim
        for i in range(num_layers):
            layer = DenseLayer(self.output_dim, growth_rate, dropout_rate)
            self.layers.append(layer)
            self.output_dim += growth_rate  # Update output dimension

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BridgeDefectClassifier(PreTrainedModel):
    def __init__(self, config):
        super(BridgeDefectClassifier, self).__init__(config)

        self.bert = BertModel.from_pretrained(config.bert_path)
        self.projected_dim = config.projected_dim

        numerical_layer_size = 32

        self.defect_number_fc = nn.Sequential(
            nn.Linear(2, numerical_layer_size),
            nn.BatchNorm1d(numerical_layer_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        self.defect_dimension_numerical_fc = nn.Sequential(
            nn.Linear(4, numerical_layer_size),
            nn.BatchNorm1d(numerical_layer_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.projected_dim),
            nn.Dropout(config.dropout_rate)
        )
        self.attention_pooling_defect_description = AttentionPooling(self.bert.config.hidden_size)
        self.attention_pooling_component = AttentionPooling(self.bert.config.hidden_size)

        input_dim = 2 * self.projected_dim + 2 * numerical_layer_size
        # Encoder part (dimension expansion)
        self.encoder_block = DenseBlock(num_layers=config.num_layers, input_dim=input_dim, growth_rate=config.growth_rate, dropout_rate=config.dropout_rate)
        encoder_output_dim = self.encoder_block.output_dim

        # Decoder part (dimension reduction)
        self.decoder_layers = nn.ModuleList()
        current_dim = encoder_output_dim
        for tl in config.transition_layers:
            layer = nn.Sequential(
                nn.BatchNorm1d(current_dim),
                nn.Linear(current_dim, tl),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.decoder_layers.append(layer)
            current_dim = tl

        # Final output layer
        self.fc = nn.Linear(current_dim, config.num_classes)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = kwargs.pop("config", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)
        return model

    def forward(
        self,
        input_ids_defect_description,
        attention_mask_defect_description,
        input_ids_component,
        attention_mask_component,
        defect_number,
        defect_dimension_numerical,
        idx=None,
        labels=None  # Add labels parameter
    ):
        # Text feature processing
        defect_description_outputs = self.bert(input_ids=input_ids_defect_description, attention_mask=attention_mask_defect_description)
        defect_description_pooled = self.attention_pooling_defect_description(defect_description_outputs.last_hidden_state, attention_mask_defect_description)
        defect_description_pooled = self.feature_projection(defect_description_pooled)

        component_outputs = self.bert(input_ids=input_ids_component, attention_mask=attention_mask_component)
        component_pooled = self.attention_pooling_component(component_outputs.last_hidden_state, attention_mask_component)
        component_pooled = self.feature_projection(component_pooled)

        # Numerical feature processing
        defect_number_outputs = self.defect_number_fc(defect_number)
        defect_dimensions_numerical_outputs = self.defect_dimension_numerical_fc(defect_dimension_numerical)

        # Merge all features
        features = torch.cat([
            defect_description_pooled,
            component_pooled,
            defect_number_outputs,
            defect_dimensions_numerical_outputs
        ], dim=1)

        # Encoder part (dimension expansion)
        features = self.encoder_block(features)  # (batch_size, encoder_output_dim)

        # Decoder part
        for layer in self.decoder_layers:
            residual = features  # Save input
            features = layer(features)
            if residual.size(1) != features.size(1):  # Check if dimensions match
                # Adjust the dimension and move to the same device
                residual = nn.Linear(residual.size(1), features.size(1)).to(features.device)(residual)
            features += residual  # Add residual connection

        # Final output layer
        logits = self.fc(features)

        outputs = {'logits': logits}

        if labels is not None:
            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss.unsqueeze(0)

        return outputs
