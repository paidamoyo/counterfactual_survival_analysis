import torch.nn as nn


class DecoderWeibull(nn.Module):
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

        self.out_zero_logscale = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.out_zero_logshape = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        self.fc_one = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=epsilon, momentum=alpha),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        self.out_one_logscale = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.out_one_logshape = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        self.reset_parameters(model_init)

    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, r0, r1):
        logshape_zero, logscale_zero = self.get_params(r=r0, fc=self.fc_zero, out_logscale=self.out_zero_logscale,
                                                       out_logshape=self.out_zero_logshape)

        logshape_one, logscale_one = self.get_params(r=r1, fc=self.fc_one, out_logscale=self.out_one_logscale,
                                                     out_logshape=self.out_one_logshape)

        t = {"logscale_zero": logscale_zero, "logshape_zero": logshape_zero, "logscale_one": logscale_one,
             "logshape_one": logshape_one}

        return t

    def get_params(self, r, fc, out_logscale, out_logshape):
        hidden_z = fc(r)
        logscale = out_logscale(hidden_z)
        logshape = out_logshape(hidden_z)
        return logshape, logscale
