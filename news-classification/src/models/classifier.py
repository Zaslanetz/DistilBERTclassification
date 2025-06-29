# src/models/classifier.py
import torch.nn as nn
from transformers import AutoModel

class NewsClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.1):
        super(NewsClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Get the hidden states from the transformer
        hidden_states = outputs.last_hidden_state
        # We take the representation of the [CLS] token (the first token)
        cls_token = hidden_states[:, 0, :]
        # Apply dropout
        cls_token = self.dropout(cls_token)
        # Feed to the classifier layer
        logits = self.classifier(cls_token)
        return logits