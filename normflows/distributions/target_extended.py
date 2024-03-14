import torch
from torch.distributions import Normal
from .target import *


class NealsFunnel(Target):
    """
    Bimodal two-dimensional distribution

    Parameters:
    prop_scale (float, optional): Scale for the distribution. Default is 20.
    prop_shift (float, optional): Shift for the distribution. Default is -10.
    v1shift (float, optional): Shift parameter for v1. Default is 0.
    v2shift (float, optional): Shift parameter for v2. Default is 0.
    """

    def __init__(self, prop_scale=torch.tensor(20.), prop_shift=torch.tensor(-10.), v1shift = 0., v2shift = 0.):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.
        self.v1shift = v1shift
        self.v2shift = v2shift
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)


    def log_prob(self, z):
        """
        Compute the log probability of the distribution for z
        The log probability of the funnel for a given pair (v, x) is defined as:

        log_prob(v, x) = log_prob_v + log_prob_x
                       = N(v | 0, 1 + v1shift) + N(x | 0, exp(0.5 * v) + v2shift)

        where:
        - N(a | b, c) denotes the log probability density of `a` under a normal distribution
          with mean `b` and standard deviation `c`.
        
        Parameters:
        z (Tensor): Value or batch of latent variable

        Returns:
        Tensor: Log probability of the distribution for z
        """
        v = z[:,0].cpu()
        x = z[:,1].cpu()
        v_like = Normal(torch.tensor([0.0]).cpu(), torch.tensor([1.0]).cpu() + self.v1shift).log_prob(v).cpu()
        x_like = Normal(torch.tensor([0.0]).cpu(), torch.exp(0.5*v).cpu() + self.v2shift).log_prob(x).cpu()
        return v_like + x_like

