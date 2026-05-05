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
                 nhead: int = 8, dropout: float = 0.1, num_layers: int = 2) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        feature_dim = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()

        self.positional_encoding = nn.Parameter(torch.randn(1, num_frames, feature_dim))

        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=nhead, # 8 têtes d'attention
            dim_feedforward=1024, 
            dropout=dropout,
            batch_first=True # Important car notre tenseur est (B, T, C)
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

        # Add positional encoding
        sequence_features += self.positional_encoding[:, :num_frames]

        # Transformer encoder: (B, T, 512)
        transformer_out = self.transformer_encoder(sequence_features)

        # Mean pooling over time: (B, 512)
        pooled_features = transformer_out.mean(dim=1)

        # Class scores: (B, num_classes)
        logits = self.classifier(pooled_features)
        return logits