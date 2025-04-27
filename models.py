import torch
import torch.nn as nn

class ChessEvalMultiTaskTransformer(nn.Module):
    def __init__(self, vocab_size=31, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        seq_len = 64 + 1 + 1  # 64 squares + 1 side-to-move + 1 castling info
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model*4, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Two heads
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 1)
        )
        self.win_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: (B, 66)
        B, L = x.size()

        tok_emb = self.embed(x)                    # (B, 66, d_model)
        pos = torch.arange(L, device=x.device)
        pos_emb = self.pos_embed(pos).unsqueeze(0) # (1, 66, d_model)

        h = tok_emb + pos_emb                      # (B, 66, d_model)
        h = self.transformer(h.permute(1, 0, 2))    # (66, B, d_model)
        h = h.mean(dim=0)                           # (B, d_model)
        h = self.norm(h)

        eval_pred = torch.tanh(self.reg_head(h)).squeeze(-1)  # (B,)
        win_pred = self.win_head(h).squeeze(-1)               # (B,) — raw logits (no activation)

        return eval_pred, win_pred
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# Resnet-Like
class ChessEvalResNet(nn.Module):
    def __init__(self, input_planes=17, channels=64, num_blocks=4):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Two heads
        self.eval_head = nn.Linear(channels, 1)  # for regression (evaluation)
        self.win_head = nn.Linear(channels, 1)   # for binary classification (winning)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.residual_blocks(x)
        x = self.global_pool(x).view(x.size(0), -1)  # shape: (batch_size, channels)

        eval_pred = self.eval_head(x)   # output shape: (batch_size, 1)
        win_pred = self.win_head(x)     # output shape: (batch_size, 1)

        return eval_pred, win_pred