import torch
from torch import nn
from torch.nn import functional as F


class VisualResidualAdapter(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout, reduction=4):
        super().__init__()
        bottleneck_dim = max(embed_dim // reduction, 1)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.normalization = nn.LayerNorm(embed_dim)
        self.down_projection = nn.Linear(embed_dim, bottleneck_dim)
        self.up_projection = nn.Linear(bottleneck_dim, embed_dim)
        self.feature_gate = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        residual = self.input_projection(features.float())
        normalized = self.normalization(residual)
        delta = self.up_projection(
            self.dropout(F.gelu(self.down_projection(normalized)))
        )
        return residual + torch.sigmoid(self.feature_gate(normalized)) * delta


class MTMProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_adapter = VisualResidualAdapter(
            config.INPUT_DIM, config.EMBED_DIM, config.DROPOUT
        )
        layers = []
        for _ in range(max(int(config.NUM_FC_LAYERS) - 1, 0)):
            layers.extend(
                [
                    nn.Linear(config.EMBED_DIM, config.EMBED_DIM),
                    nn.ReLU(inplace=True),
                    nn.Dropout(config.DROPOUT),
                ]
            )
        self.feature_projection = nn.Sequential(*layers)
        attention_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.EMBED_DIM * 4,
            dropout=config.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.self_attention = nn.TransformerEncoder(
            attention_layer, num_layers=config.NUM_ATTENTION_LAYERS
        )
        self.output_normalization = nn.LayerNorm(config.EMBED_DIM)

    def forward(self, raw_features, use_visual_adapter=True):
        visual_features = (
            self.visual_adapter(raw_features)
            if use_visual_adapter
            else raw_features.float()
        )
        projected = self.feature_projection(visual_features).unsqueeze(0)
        projected = self.output_normalization(
            self.self_attention(projected).squeeze(0)
        )
        return visual_features, projected
