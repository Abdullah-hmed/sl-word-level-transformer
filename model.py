import torch
import torch.nn as nn

class SignLanguageTransformer(nn.Module):
    def __init__(self, input_dim=180*3, embed_dim=512, num_heads=16, num_classes=30, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths):
        B, T, J, C = x.shape  # (Batch, Time, 180, 3)
        x = x.reshape(B, T, -1)  # Flatten joints: (B, T, 180*3)
        x = self.embedding(x)  # (B, T, embed_dim)
        x = self.dropout(x)

        # Create attention mask (True = ignore, False = keep)
        max_len = T
        mask = torch.arange(max_len, device=x.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

        # Apply Transformer with mask
        x = self.transformer(x, src_key_padding_mask=mask)  # (B, T, embed_dim)

        # Global pooling approach - average over time
        # Create a mask for padding (0 for padded positions, 1 for actual data)
        seq_mask = (~mask).float().unsqueeze(-1)  # (B, T, 1)
        
        # Apply mask and compute mean
        x = (x * seq_mask).sum(dim=1) / lengths.unsqueeze(-1).float()

        return self.fc(x)  # Classification layer
