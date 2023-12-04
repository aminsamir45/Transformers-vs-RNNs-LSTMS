# Transformer FFN Embedding Model with Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

D_EMBED = 128
HIDDEN = 256
P = 113
HEADS = 2

class FFNAttention(nn.Module):
    def __init__(self):
        super(FFNAttention, self).__init__()
        self.embed = nn.Embedding(P, D_EMBED)
        self.attention = nn.MultiheadAttention(embed_dim=2*D_EMBED, num_heads=HEADS)
        self.linear1 = nn.Linear(2 * D_EMBED, HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, P)
        self.init_weights()
        
    def forward(self, x1, x2):
        x1 = self.embed(x1)
        x2 = self.embed(x2)
        x = torch.cat((x1, x2), dim=1).unsqueeze(0)  # Adding batch dimension for attention
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)  # Removing batch dimension after attention
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)               
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)