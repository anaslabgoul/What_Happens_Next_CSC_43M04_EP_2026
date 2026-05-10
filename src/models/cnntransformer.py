"""
Forward pass of the model:
    Input:  (batch, time, C, H, W)
    Reshape: (batch * time, C, H, W)  # each frame is an independent image
    Backbone: ResNet18 up to global average pool -> (batch * time, 512, 1, 1)
    Flatten: (batch * time, 512)
    Reshape: (batch, time, 512)
    Mean over time: (batch, 512)
    Linear classifier: (batch, num_classes)
"""


import torch
import torch.nn as nn
import torchvision.models as models


class CNNTransformer(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False, num_frames: int = 4,
                 nhead: int = 2, dropout: float = 0.5, num_layers: int = 1) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        feature_dim = backbone.fc.in_features  # 512 for ResNet18
        transformer_dim = 128
        backbone.fc = nn.Identity()

        self.proj = nn.Linear(feature_dim, transformer_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_frames, transformer_dim))

        self.backbone = backbone
        self.fc_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(transformer_dim, num_classes)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,  # Reduced heads for a lighter temporal encoder
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True  # Input shape is (B, T, C)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (batch_size, T, C, H, W)
        returns logits: (batch_size, num_classes)
        """
        batch_size, num_frames, channels, height, width = video_batch.shape

        # Merge batch and time so la CNN runs frame-wise: (B*T, C, H, W)
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        # (B*T, 512, 1, 1) -> (B*T, 512)
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        # Restore temporal structure: (B, T, 512)
        sequence_features = frame_features.view(batch_size, num_frames, -1)
        sequence_features = self.proj(sequence_features)  # (B, T, 64)

        # Add positional encoding
        sequence_features += self.positional_encoding[:, :num_frames]

        # Transformer encoder: (B, T, 64)
        transformer_out = self.transformer_encoder(sequence_features)

        # Mean pooling over time: (B, 64)
        pooled_features = transformer_out.mean(dim=1)

        pooled_features = self.fc_dropout(pooled_features)

        # Class scores: (B, num_classes)
        logits = self.classifier(pooled_features)
        return logits