import torch
import torch.nn as nn

# ----------------------------
# ResNet blocks (1D)
# ----------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class ECGResNet(nn.Module):
    """
    Returns feature map [B, 512, T’] for BiLSTM input
    """
    def __init__(self, input_channels=12, layers=(2, 2, 2, 2)):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15,
                               stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [ResidualBlock1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 12, T]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, 512, T’]


def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


class ECG_CNN_BiLSTM(nn.Module):
    def __init__(
        self,
        cnn_encoder: nn.Module,
        tabular_dim: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        num_classes: int = 27,  # Defaulted to 27 classes
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = cnn_encoder
        self.use_tabular = tabular_dim > 0

        # --- ECG Temporal Encoder ---
        self.bilstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.ecg_emb_dim = 2 * lstm_hidden

        # --- Conditional Tabular Branch ---
        if self.use_tabular:
            self.tabular_mlp = nn.Sequential(
                nn.Linear(tabular_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.tab_emb_dim = 64
        else:
            self.tabular_mlp = nn.Identity()
            self.tab_emb_dim = 0

        # --- Dynamic Classifier Head ---
        self.fused_dim = self.ecg_emb_dim + self.tab_emb_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes), 
        )

    def encode_ecg(self, ecg: torch.Tensor) -> torch.Tensor:
        # CNN feature map
        x = self.cnn(ecg)          # [B, 512, T']
        x = x.permute(0, 2, 1)     # [B, T', 512]
        
        # BiLSTM Sequence
        x, _ = self.bilstm(x)      # [B, T', 2H]
        
        # POOLING UPGRADE: Combine Max and Mean pooling 
        # This captures both "overall rhythm" (mean) and "isolated ectopic beats" (max)
        avg_pool = x.mean(dim=1)   # [B, 2H]
        max_pool = x.max(dim=1)[0] # [B, 2H]
        z_ecg = avg_pool + max_pool 
        
        return z_ecg

    def encode_fused(self, ecg: torch.Tensor, tabular: torch.Tensor = None) -> torch.Tensor:
        z_ecg = self.encode_ecg(ecg)
        
        if self.use_tabular and tabular is not None:
            z_tab = self.tabular_mlp(tabular)
            return torch.cat([z_ecg, z_tab], dim=1) # [B, 2H + 64]
        
        return z_ecg
    
    def extract_fused(self, ecg, tab=None):
        """Froze encoder -> fused embedding (no grad)"""
        return self.encode_fused(ecg, tab)
    
    def forward_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused)

    def forward(self, ecg: torch.Tensor, tabular: torch.Tensor = None) -> torch.Tensor:
        fused = self.encode_fused(ecg, tabular)
        return self.forward_from_fused(fused)
    
