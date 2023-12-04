import torch
import torch.nn as nn
import torch.nn.functional as F

D_EMBED = 128
HIDDEN = 256
P = 59 
NUM_LAYERS = 2 
DROPOUT = 0.5

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(P, D_EMBED)
        self.lstm = nn.LSTM(input_size=D_EMBED, hidden_size=HIDDEN, 
                            num_layers=NUM_LAYERS, batch_first=True,
                            dropout=DROPOUT)
        self.linear = nn.Linear(HIDDEN, P)
        self.init_weights()

    def forward(self, x1, x2):
        x1 = self.embed(x1)
        x2 = self.embed(x2)
        x = torch.stack((x1, x2), dim=1)
        _, (h_n, _) = self.lstm(x)
        x = self.linear(h_n[-1])
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()
#         self.embed = nn.Embedding(P, D_EMBED)
#         self.rnn = nn.RNN(input_size=D_EMBED, hidden_size=HIDDEN, batch_first=True)
#         self.linear = nn.Linear(HIDDEN, P)
#         self.init_weights()

#     def forward(self, x1, x2):
#         x1 = self.embed(x1)
#         x2 = self.embed(x2)
#         x = torch.stack((x1, x2), dim=1)
#         _, h_n = self.rnn(x)
#         x = self.linear(h_n.squeeze(0))
#         return x

#     # Weight initialization
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Embedding):
#                 nn.init.xavier_normal_(m.weight)
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 nn.init.zeros_(m.bias)