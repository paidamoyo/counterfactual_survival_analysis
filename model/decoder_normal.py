import torch.nn as nn


class DecoderNormal(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout, model_init):
        nn.Module.__init__(self)
        epsilon = 1e-5
        alpha = 0.9  # use numbers closer to 1 if you have more data
        self.hidden_dim = hidden_dim
        self.fc_zero = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=epsilon, momentum=alpha),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.out_zero_mu = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.out_zero_log_var = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        self.fc_one = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=epsilon, momentum=alpha),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        self.out_one_mu = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.out_one_log_var = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        self.reset_parameters(model_init)

    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, r0, r1):
        logvar_zero, mu_zero = self.get_params(r=r0, fc=self.fc_zero, out_mu=self.out_zero_mu,
                                               out_log_var=self.out_zero_log_var)

        logvar_one, mu_one = self.get_params(r=r1, fc=self.fc_one, out_mu=self.out_one_mu,
                                             out_log_var=self.out_one_log_var)

        t = {"mu_zero": mu_zero, "logvar_zero": logvar_zero, "mu_one": mu_one, "logvar_one": logvar_one}

        return t

    def get_params(self, r, fc, out_mu, out_log_var):
        hidden_z = fc(r)
        mu = out_mu(hidden_z)
        logvar = out_log_var(hidden_z)
        return logvar, mu
