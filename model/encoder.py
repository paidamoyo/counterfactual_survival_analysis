import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, model_init):
        nn.Module.__init__(self)
        epsilon = 1e-5
        self.hidden_dim = hidden_dim
        alpha = 0.9  # use numbers closer to 1 if you have more data
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=epsilon, momentum=alpha),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=epsilon, momentum=alpha),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.reset_parameters(model_init)

    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, x):
        h_1 = self.fc_1(x)
        r = self.fc_2(h_1)
        return r