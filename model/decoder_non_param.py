import torch.nn as nn
from torch.distributions import uniform
import torch
from torch.nn.modules import Module


class Exp(Module):
    r"""Applies the element-wise function :math:`\text{Exp}(x) = exp(x)}`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input


    Examples::

        >>> m = nn.Exp()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input):
        return torch.exp(input)


class DecoderNonParam(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout, model_init, is_stochastic):
        nn.Module.__init__(self)
        epsilon = 1e-5
        alpha = 0.9  # use numbers closer to 1 if you have more data
        self.hidden_dim = hidden_dim
        self.is_stochastic = is_stochastic
        self.fc_zero = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=epsilon, momentum=alpha),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.fc_one = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=epsilon, momentum=alpha),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        if is_stochastic:
            fc_hidden_dim = hidden_dim * 2
            self.t_noise_zero = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
                nn.Tanh()

            )

            self.t_noise_one = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
                nn.Tanh()
            )

            self.c_noise_zero = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
                nn.Tanh()

            )

            self.c_noise_one = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
                nn.Tanh()
            )
        else:
            fc_hidden_dim = hidden_dim

        self.t_out_zero_mu = nn.Sequential(
            nn.Linear(in_features=fc_hidden_dim, out_features=output_dim),
            Exp()
        )
        self.t_out_one_mu = nn.Sequential(
            nn.Linear(in_features=fc_hidden_dim, out_features=output_dim),
            Exp()
        )

        self.c_out_zero_mu = nn.Sequential(
            nn.Linear(in_features=fc_hidden_dim, out_features=output_dim),
            Exp()
        )
        self.c_out_one_mu = nn.Sequential(
            nn.Linear(in_features=fc_hidden_dim, out_features=output_dim),
            Exp()
        )

        self.reset_parameters(model_init)

    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, r0, r1):
        if self.is_stochastic:
            t_mu_zero = self.get_params_stoch(r=r0, fc=self.fc_zero, out_mu=self.t_out_zero_mu,
                                              noise_trans=self.t_noise_zero)
            t_mu_one = self.get_params_stoch(r=r1, fc=self.fc_one, out_mu=self.t_out_one_mu,
                                             noise_trans=self.t_noise_one)

            c_mu_zero = self.get_params_stoch(r=r0, fc=self.fc_zero, out_mu=self.c_out_zero_mu,
                                              noise_trans=self.c_noise_zero)
            c_mu_one = self.get_params_stoch(r=r1, fc=self.fc_one, out_mu=self.c_out_one_mu,
                                             noise_trans=self.c_noise_one)
        else:
            t_mu_zero = self.get_params(r=r0, fc=self.fc_zero, out_mu=self.t_out_zero_mu)
            t_mu_one = self.get_params(r=r1, fc=self.fc_one, out_mu=self.t_out_one_mu)

            c_mu_zero = self.get_params(r=r0, fc=self.fc_zero, out_mu=self.c_out_zero_mu)
            c_mu_one = self.get_params(r=r1, fc=self.fc_one, out_mu=self.c_out_one_mu)

        t = {"t_mu_zero": t_mu_zero, "t_mu_one": t_mu_one, "c_mu_zero": c_mu_zero,
             "c_mu_one": c_mu_one}

        return t

    def get_params(self, r, fc, out_mu):
        hidden_r = fc(r)
        mu = out_mu(hidden_r)
        return mu

    def get_params_stoch(self, r, fc, out_mu, noise_trans):
        hidden_r = fc(r)
        batch_size = r.shape[0]
        epsilon = uniform.Uniform(low=0, high=1).sample(
            sample_shape=torch.Size([batch_size, self.hidden_dim]))
        epsilon_trans = epsilon + noise_trans(epsilon)
        hidden_noise = torch.cat((hidden_r, epsilon_trans), dim=1)
        mu = out_mu(hidden_noise)
        return mu