import torch
import torch.nn as nn
from models.encoder_ECG import ECGResNet

class ECGClassifier(nn.Module):
    def __init__(self, ecg_leads=12, num_classes=4):
        super(ECGClassifier, self).__init__()
        self.ecg_encoder = ECGResNet(input_channels=ecg_leads)
        self.fc = nn.Linear(512, num_classes)  

    def forward(self, x):
        features = self.ecg_encoder(x)  # Shape: [batch_size, 512]
        logits = self.fc(features)       # Shape: [batch_size, num_classes] 
        return logits