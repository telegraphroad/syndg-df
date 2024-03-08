import math
from numbers import Number

import numpy as np
import pyro.distributions as dist
import torch
import torch.nn as nn
from scipy import stats
from torch.distributions import (Categorical, Distribution, MixtureSameFamily,
                                 Normal, StudentT, constraints)
from torch.distributions.utils import broadcast_all
from ..utils.preprocessing import CSVDataset

from .base import BaseDistribution


# First we define the following constands to improve performance by reducing computational overhead, 
# enhance accuracy by avoiding repeated recalculations in floating-point arithmetic, and increase 
# code readability and maintainability through clear, descriptive identifiers.

# CONST_SQRT_2: Square root of 2. This constant is commonly used in various statistical formulas,
# e.g. the normalization a Gaussian where factors of sqrt(2) arise in the denominator of exponential 
# terms or in the computation of standard deviations and variances.
CONST_SQRT_2 = math.sqrt(2)

# CONST_INV_SQRT_2PI: The multiplicative inverse of the square root of 2π. This constant is a critical
# factor in the normalization constant of the PDF for the Gaussian
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)

# CONST_INV_SQRT_2: The multiplicative inverse of the square root of 2, used in transformations involving 
# the Gaussian distribution, e.g. converting from a standard Gaussian to other forms
# or in the implementation of certain statistical tests and confidence interval calculations
#useful for the 2nd postdoc project
CONST_INV_SQRT_2 = 1 / math.sqrt(2)

# CONST_LOG_INV_SQRT_2PI: The log of CONST_INV_SQRT_2PI. Logging this value is useful for
# computations where the logarithm of the normalization constant is required directly, e.g. in the calculation
# of log-likelihoods for Gaussians.
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)

# CONST_LOG_SQRT_2PI_E: Half of the natural logarithm of 2πe. This appears in entropy calculations
# for normal distributions. Euler's number often arises in contexts involving growth
# processes or natural processes described by exponential functions.
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

class GeneralizedGaussianDistribution(Distribution):
    """
    Generalized Gaussian Distribution (GGD)
    https://en.wikipedia.org/wiki/Generalized_normal_distribution

    This class is a PyTorch implementation of the Generalized Gaussian distribution,
    which includes both symmetric and asymmetric versions. The GGD is characterized by 
    a shape parameter that allows the distribution to represent both platykurtic (flat-topped)
    and leptokurtic (peaked) distributions, making it a flexible model for various kinds of data.

    The distribution's probability density function (pdf) is given by:
    
    f(x | μ, α, β) = β / (2αΓ(1/β)) * exp(-(|x - μ| / α) ^ β)
    
    where:
    - μ (loc) is the location parameter, which shifts the distribution along the x-axis.
    - α (scale) is the scale parameter, which stretches or compresses the distribution.
    - β (p) is the shape parameter, which controls the peakedness and tail behavior of the distribution.

    Parameters:
    loc (Tensor): the location parameter of the distribution.
    scale (Tensor): the scale parameter of the distribution.
    p (Tensor): the shape parameter of the distribution.
    validate_args (bool, optional): checks if the arguments are valid. Default is None.

    """
    
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'p': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, p, validate_args=None):
        # Broadcast all inputs to ensure they have the same shape
        self.loc, self.scale = broadcast_all(loc, scale)
        (self.p,) = broadcast_all(p)
        
        # Convert tensor to numpy for scipy compatibility
        self.scipy_dist = stats.gennorm(loc=self.loc.cpu().detach().numpy(),
                            scale=self.scale.cpu().detach().numpy(),
                            beta=self.p.cpu().detach().numpy())
        
        # Determine batch shape based on input types
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
            
        # Initialize the base class
        super(GeneralizedGaussianDistribution, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        # Mean of the distribution is its location parameter
        return self.loc

    @property
    def variance(self):
        # Variance is calculated using the scale and shape parameters
        # The formula for variance is given by:
        # variance = scale^2 * (exp(lgamma(3/p) - lgamma(1/p)))
        return self.scale.pow(2) * (torch.lgamma(3/self.p) - torch.lgamma(1/self.p)).exp()

    @property
    def stddev(self):
        # Standard deviation is the square root of variance
        return self.variance**0.5

    def expand(self, batch_shape, _instance=None):
        # Create a new instance of the distribution with expanded batch shape
        new = self._get_checked_instance(GeneralizedGaussianDistribution, _instance)
        # Ensure the batch shape is a proper torch.Size object for compatibility.

        batch_shape = torch.Size(batch_shape)
        # Expand the 'loc' and 'scale' parameters to the new batch shape without copying data.
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        # Initialize the new instance with the expanded batch shape, skipping validation for efficiency.        
        super(GeneralizedGaussianDistribution, new).__init__(batch_shape, validate_args=False)
        # Preserve the validation flag from the original instance to the new one.
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        # Generate a sample from the distribution
        sample_shape = sample_shape + self.loc.size()
        return torch.tensor(self.scipy_dist.rvs(
            list(sample_shape),
            random_state=torch.randint(2**32, ()).item()),  # Make deterministic if torch is seeded
                            dtype=self.loc.dtype, device=self.loc.device)

    def log_prob(self, value):
        # Calculate the log probability density function (log-PDF) of the Generalized Gaussian Distribution.
        # The formula for the log-PDF is:
        # log(P(x)) = -log(2 * scale) - Γ(1/p) + log(p) - ((|x - loc| / scale)^p)
        # where:
        # x is the input value,
        # loc is the location parameter of the distribution,
        # scale is the scale parameter of the distribution,
        # p is the shape parameter of the distribution,
        # Γ(.) is the gamma function,
        # |x - loc| is the absolute difference between x and the location parameter,
        # and ^p denotes raising to the power p.
        if self._validate_args:
            self._validate_sample(value)
        return (-torch.log(2 * self.scale) - torch.lgamma(1/self.p) + torch.log(self.p)
                - torch.pow((torch.abs(value - self.loc) / self.scale), self.p))

    def cdf(self, value):
        # Compute the cumulative distribution function at a given value
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        return torch.tensor(self.scipy_dist.cdf(value),
                            dtype=self.loc.dtype, device=self.loc.device)

    def icdf(self, value):
        # Inverse cumulative distribution function is not implemented
        raise NotImplementedError

    def entropy(self):
        # Compute the entropy of the distribution using the formula:
        # H(X) = 1/β - log(β) + log(2α) + Γ(1/β),
        # where α is the scale, β is the shape parameter, and Γ is the gamma function.
        #
        # The formula combines these components to yield the entropy, a measure of uncertainty
        # or randomness. The entropy is higher when the distribution is more spread out
        # (higher α) and has heavier tails (lower β).

        return (1/self.p) - torch.log(self.p) + torch.log(2*self.scale) + torch.lgamma(1/self.p)
        



class GeneralizedGaussianMixture(BaseDistribution):
    """
    Mixture of Generalized Gaussians

    This class represents a mixture model composed of Generalized Gaussian distributions. 
    Each component of the mixture is a Generalized Gaussian distribution, which extends the 
    standard Gaussian distribution by allowing different shapes of the probability density 
    function, controlled by the shape parameter `p`.

    The probability density function of a Generalized Gaussian distribution is given by:
    
    ```
    f(x | mu, alpha, p) = p / (2 * alpha * Gamma(1/p)) * exp(-(|x - mu| / alpha)^p)
    ```
    
    where:
    - `x` represents the data points,
    - `mu` is the mean (location parameter),
    - `alpha` is the scale parameter related to the spread of the distribution,
    - `p` is the shape parameter which dictates the shape of the distribution,
    - `Gamma` is the gamma function.

    The mixture model's density function is then:
    
    ```
    f(x) = sum(w_k * f_k(x | mu_k, alpha_k, p_k) for k in range(1, K+1))
    ```
    
    where:
    - `K` is the number of modes in the mixture,
    - `w_k` are the weights of each mode, satisfying sum(w_k) = 1,
    - `f_k` denotes the density function of the k-th Generalized Gaussian component.

    Parameters:
    n_modes (int): Number of modes of the mixture model
    dim (int): Number of dimensions of each Gaussian
    loc (float, optional): Mean values. Default is 0.
    scale (float, optional): Diagonals of the covariance matrices. Default is 1.
    p (float, optional): Shape parameter for the Generalized Gaussian. Default is 2.
    rand_p (bool, optional): If True, shape parameter p is randomized. Default is True.
    noise_scale (float, optional): Scale of the noise added to shape parameter p. Default is 0.01.
    weights (list, optional): List of mode probabilities. Default is None.
    trainable_loc (bool, optional): If True, location parameters will be optimized during training. Default is False.
    trainable_scale (bool, optional): If True, scale parameters will be optimized during training. Default is True.
    trainable_p (bool, optional): If True, shape parameters will be optimized during training. Default is True.
    trainable_weights (bool, optional): If True, weights will be optimized during training. Default is True.
    ds (CSVDataset, optional): Dataset object used to initialize the location parameters. Default is None.
    device (str, optional): Device to which tensors will be moved. Default is 'cuda'.
    """

    def __init__(self, n_modes, dim, loc=0., scale=1., p=2., rand_p=True, noise_scale=0.01, weights=None, trainable_loc=True, trainable_scale=True, trainable_p=True, trainable_weights=True, ds:CSVDataset=None, device='cuda'):
        super().__init__()
        with torch.no_grad():
            self.n_modes = n_modes
            self.dim = dim
            self.device = device

            # Initialize location, scale and shape parameters
            if ds is None:
                loc = np.zeros((self.n_modes, self.dim)) + loc
            else:
                loc = np.tile(ds.calculate_feature_means(),(n_modes,1)) + loc
            scale = np.zeros((self.n_modes, self.dim)) + scale
            p = np.zeros((self.n_modes, self.dim)) + p
            if rand_p:
                noise = np.random.normal(0, noise_scale, p.shape)
                p += noise
                loc += noise
                scale += np.abs(noise)
                
            # Initialize weights
            if weights is None:
                weights = np.ones(self.n_modes)
            weights /= np.sum(weights)

            # Create parameters or buffers depending on whether they are trainable or not
            if trainable_loc:
                self.loc = nn.Parameter(torch.tensor(1.0 * loc, device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("loc", torch.tensor(1.0 * loc, device=self.device).float())
            if trainable_scale:
                self.scale = nn.Parameter(torch.tensor(1.0 * scale, device=self.device).float(),requires_grad=True) 
            else:
                self.register_buffer("scale", torch.tensor(1.0 * scale, device=self.device).float())
            if trainable_p:
                self.p = nn.Parameter(torch.tensor(1.0 * p, device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("p", torch.tensor(1.0 * p, device=self.device).float())
            if trainable_weights:
                self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights), device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights), device=self.device).float())
            # Initialize the underlying Generalized Gaussian and Categorical distributions
            self.gg = torch.distributions.Independent(GeneralizedGaussianDistribution(self.loc, self.scale, self.p),1)
            self.cat = Categorical(torch.softmax(self.weight_scores, 0))

            self.mixture = MixtureSameFamily(self.cat, self.gg)


    def forward(self, num_samples=1):
        # Sample mode indices
        z = self.mixture.sample((num_samples,))

        # Compute log probability
        log_p = self.mixture.log_prob(z)

        return z, log_p
    
    def log_prob(self, z):
        # Compute log probability
        log_p = self.mixture.log_prob(z)
        return log_p
    

    
class GaussianMixture(BaseDistribution):
    """
    Mixture of  Gaussians

    Parameters:
    n_modes (int): Number of modes of the mixture model
    dim (int): Number of dimensions of each Gaussian
    loc (float, optional): Mean values. Default is 0.
    scale (float, optional): Diagonals of the covariance matrices. Default is 1.
    p (float, optional): Shape parameter for the Generalized Gaussian. Default is 2.
    rand_p (bool, optional): If True, shape parameter p is randomized. Default is True.
    noise_scale (float, optional): Scale of the noise added to shape parameter p. Default is 0.01.
    weights (list, optional): List of mode probabilities. Default is None.
    trainable_loc (bool, optional): If True, location parameters will be optimized during training. Default is False.
    trainable_scale (bool, optional): If True, scale parameters will be optimized during training. Default is True.
    trainable_p (bool, optional): If True, shape parameters will be optimized during training. Default is True.
    trainable_weights (bool, optional): If True, weights will be optimized during training. Default is True.
    ds (CSVDataset, optional): Dataset object used to initialize the location parameters. Default is None.
    device (str, optional): Device to which tensors will be moved. Default is 'cuda'.
    """

    def __init__(self, n_modes, dim, loc=0., scale=1., p=2., rand_p=True, noise_scale=0.01, weights=None, trainable_loc=True, trainable_scale=True, trainable_p=True, trainable_weights=True, ds=None, device='cuda'):
        super().__init__()
        with torch.no_grad():
            self.n_modes = n_modes
            self.dim = dim
            self.device = device

            # Initialize location, scale and shape parameters
            if ds is None:
                loc = np.zeros((self.n_modes, self.dim)) + loc
            else:
                loc = np.tile(ds.calculate_feature_means(),(n_modes,1)) + loc
            scale = np.zeros((self.n_modes, self.dim)) + scale
            p = np.zeros((self.n_modes, self.dim)) + p
            if rand_p:
                noise = np.random.normal(0, noise_scale, p.shape)
                p += noise
                loc += noise
                scale += np.abs(noise)
                
            # Initialize weights
            if weights is None:
                weights = np.ones(self.n_modes)
            weights /= np.sum(weights)

            # Create parameters or buffers depending on whether they are trainable or not
            if trainable_loc:
                self.loc = nn.Parameter(torch.tensor(1.0 * loc, device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("loc", torch.tensor(1.0 * loc, device=self.device).float())
            if trainable_scale:
                self.scale = nn.Parameter(torch.tensor(1.0 * scale, device=self.device).float(),requires_grad=True) 
            else:
                self.register_buffer("scale", torch.tensor(1.0 * scale, device=self.device).float())
            if trainable_p:
                self.p = nn.Parameter(torch.tensor(1.0 * p, device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("p", torch.tensor(1.0 * p, device=self.device).float())
            if trainable_weights:
                self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights), device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights), device=self.device).float())
            # Initialize the underlying Generalized Gaussian and Categorical distributions
            self.gg = torch.distributions.Independent(torch.distributions.Normal(self.loc, self.scale),1)
            self.cat = Categorical(torch.softmax(self.weight_scores, 0))

            self.mixture = MixtureSameFamily(self.cat, self.gg)


    def forward(self, num_samples=1):
        # Sample mode indices
        z = self.mixture.sample((num_samples,))

        # Compute log probability
        log_p = self.mixture.log_prob(z)

        return z, log_p
    
    def log_prob(self, z):
        # Compute log probability
        log_p = self.mixture.log_prob(z)
        return log_p
    

class StudentTDistribution(BaseDistribution):
    """
    A wrapper class for the Student's t-distribution, allowing for the creation of a distribution
    with trainable or fixed parameters for location, scale, and degrees of freedom.

    The probability density function (PDF) for the Student's t-distribution is given by:
    
        f(x | nu) = (Gamma((nu + 1) / 2) / (sqrt(nu * pi) * Gamma(nu / 2))) * (1 + x^2 / nu)^(-(nu + 1) / 2)
    
    where `nu` is the degrees of freedom, `x` is the variable, and `Gamma` is the gamma function.

    Attributes:
        shape (tuple): The shape of the distribution.
        n_dim (int): Number of dimensions of the distribution.
        loc (torch.Tensor): Location parameter of the distribution.
        log_scale (torch.Tensor): Logarithm of the scale parameter of the distribution.
        df (torch.Tensor): Degrees of freedom parameter of the distribution.

    Parameters:
        shape (int or tuple): Specifies the shape of the batch dimensions.
        df (float, optional): Initial value for the degrees of freedom. Default is 2.0.
        trainable (bool, optional): If True, parameters will be trainable. Default is True.
        device (str, optional): Device to which parameters are assigned. Default is 'cuda'.
    """
    def __init__(self, shape, df=2.0, trainable=True, device='cuda'):
        super().__init__()
        # Convert shape input to a tuple if it isn't already, this was a problem with the newer driver code
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.device = device

        # Create parameters or buffers depending on whether they are trainable or not
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape, device=self.device))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape, device=self.device))
            self.df = nn.Parameter(torch.tensor(df,device=self.device))  # degrees of freedom
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
            self.register_buffer("df", torch.tensor(df))

    def forward(self, num_samples=1):
        """
        Samples from the Student's t-distribution and computes the log probability of the sampled values.

        Parameters:
            num_samples (int, optional): Number of samples to generate. Default is 1.

        Returns:
            tuple: A tuple containing:
                - sampled values (torch.Tensor)
                - log probabilities of these values (torch.Tensor)
        """        
        z = StudentT(self.df, self.loc, torch.exp(self.log_scale)).sample((num_samples,))
        log_p = StudentT(self.df, self.loc, torch.exp(self.log_scale)).log_prob(z).sum(dim=1)
        return z.squeeze(), log_p

    def log_prob(self, z):
        """
        Computes the log probability of given values using the Student's t-distribution.

        Parameters:
            z (torch.Tensor): Values at which to compute the log probability.

        Returns:
            torch.Tensor: Log probabilities of the given values.
        """
        log_p = StudentT(self.df, self.loc, torch.exp(self.log_scale)).log_prob(z).sum(dim=1)
        return log_p
    

class MultivariateStudentTDist(nn.Module):
    """
    A module for the multivariate Student's t-distribution, supporting both trainable
    and non-trainable parameters for location, scale, and degrees of freedom.

    The probability density function (PDF) for the multivariate Student's t-distribution is:
    
        f(x | nu, mu, Sigma) = Gamma((nu + d) / 2) / (Gamma(nu / 2) * (nu * pi)^(d/2) * det(Sigma)^(1/2))
                               * (1 + (1/nu) * (x - mu)^T Sigma^(-1) (x - mu))^(-(nu + d) / 2)
    
    where `nu` is the degrees of freedom, `mu` is the location vector, `Sigma` is the scale matrix
    (covariance), and `d` is the dimensionality of the distribution.

    Attributes:
        dim (int): Dimensionality of the distribution.
        loc (torch.Tensor): Location vector of the distribution.
        scale_tril (torch.Tensor): Lower triangular matrix representing the Cholesky decomposition
                                   of the scale matrix.
        df (torch.Tensor): Degrees of freedom parameter of the distribution.

    Parameters:
        degree_of_freedom (float): Initial value for the degrees of freedom.
        dim (int): Dimensionality of the distribution.
        trainable (bool, optional): If True, parameters will be trainable. Default is True.
        device (str, optional): Device to which parameters are assigned. Default is 'cpu'.
    """    
    def __init__(self, degree_of_freedom, dim, trainable=True, device='cpu'):
        #for the ablation, need to add an option to trigger numerical instability, if not I'll just trust the t to be its crazy unstable self with any df below 4-5
        super().__init__()
        self.dim = dim
        self.device = device

        if trainable:
            self.loc = nn.Parameter(torch.zeros((dim,), device=self.device))
            self.scale_tril = nn.Parameter(torch.eye(self.dim, device=self.device))
            self.df = nn.Parameter(torch.tensor(degree_of_freedom, device=self.device))
        else:
            self.register_buffer("loc", torch.zeros((dim,), device=self.device))
            self.register_buffer("scale_tril", torch.eye(self.dim, device=self.device))
            self.register_buffer("df", torch.tensor(degree_of_freedom, device=self.device))

    def forward(self, num_samples):
        """
        Generates samples and calculates their log probabilities from the multivariate Student's t-distribution.

        Parameters:
            num_samples (int): Number of samples to generate.

        Returns:
            tuple: A tuple containing:
                - samples (torch.Tensor): Generated samples.
                - log_prob (torch.Tensor): Log probabilities of the generated samples.
        """        
        mvt = dist.MultivariateStudentT(self.df,self.loc,self.scale_tril)
        samples = mvt.sample((num_samples,))
        log_prob = self.log_prob(samples)
        return samples, log_prob

    def log_prob(self, samples):
        """
        Computes the log probability of given samples under the multivariate Student's t-distribution.

        Parameters:
            samples (torch.Tensor): Samples to compute the log probabilities for.

        Returns:
            torch.Tensor: Log probabilities of the samples.
        """        
        return dist.MultivariateStudentT(self.df,self.loc,self.scale_tril).log_prob(samples)
    

class StudentTMixture(BaseDistribution):
    """
    Mixture of Student's Ts

    Parameters:
    n_modes (int): Number of modes of the mixture model
    dim (int): Number of dimensions of each Gaussian
    loc (float, optional): Mean values. Default is 0.
    scale (float, optional): Diagonals of the covariance matrices. Default is 1.
    p (float, optional): Shape parameter for the Generalized Gaussian. Default is 2.
    rand_p (bool, optional): If True, shape parameter p is randomized. Default is True.
    noise_scale (float, optional): Scale of the noise added to shape parameter p. Default is 0.01.
    weights (list, optional): List of mode probabilities. Default is None.
    trainable_loc (bool, optional): If True, location parameters will be optimized during training. Default is False.
    trainable_scale (bool, optional): If True, scale parameters will be optimized during training. Default is True.
    trainable_p (bool, optional): If True, shape parameters will be optimized during training. Default is True.
    trainable_weights (bool, optional): If True, weights will be optimized during training. Default is True.
    device (str, optional): Device to which tensors will be moved. Default is 'cuda'.
    """

    def __init__(self, n_modes, dim, loc=0., scale=1., p=2., rand_p=True, noise_scale=0.01, weights=None, trainable_loc=True, trainable_scale=True, trainable_p=True, trainable_weights=True, ds=None, device='cuda'):
        super().__init__()
        with torch.no_grad():
            self.n_modes = n_modes
            self.dim = dim
            self.device = device

            # Initialize location, scale and shape parameters
            if ds is None:
                loc = np.zeros((self.n_modes, self.dim)) + loc
            else:
                loc = np.tile(ds.calculate_feature_means(),(n_modes,1)) + loc
            scale = np.zeros((self.n_modes, self.dim)) + scale
            p = np.zeros((self.n_modes, self.dim)) + p
            if rand_p:
                noise = np.random.normal(0, noise_scale, p.shape)
                p += noise
                loc += noise
                scale += np.abs(noise)
                
            # Initialize weights
            if weights is None:
                weights = np.ones(self.n_modes)
            weights /= np.sum(weights)

            # Create parameters or buffers depending on whether they are trainable or not
            if trainable_loc:
                self.loc = nn.Parameter(torch.tensor(1.0 * loc, device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("loc", torch.tensor(1.0 * loc, device=self.device).float())
            if trainable_scale:
                self.scale = nn.Parameter(torch.tensor(1.0 * scale, device=self.device).float(),requires_grad=True) 
            else:
                self.register_buffer("scale", torch.tensor(1.0 * scale, device=self.device).float())
            if trainable_p:
                self.p = nn.Parameter(torch.tensor(1.0 * p, device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("p", torch.tensor(1.0 * p, device=self.device).float())
            if trainable_weights:
                self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights), device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights), device=self.device).float())


            # Initialize the Student T distributions and the categorical distribution
            student_t_distributions = [MultivariateStudentTDist(self.df[i], dim, trainable=False, device=self.device) for i in range(self.n_modes)]
            self.cat = Categorical(logits=self.weight_scores)
            self.mixture = MixtureSameFamily(self.cat, student_t_distributions)

            # Initialize the underlying Generalized Gaussian and Categorical distributions


    def forward(self, num_samples=1):
        # Sample mode indices
        z = self.mixture.sample((num_samples,))

        # Compute log probability
        log_p = self.mixture.log_prob(z)

        return z, log_p
    
    def log_prob(self, z):
        # Compute log probability
        log_p = self.mixture.log_prob(z)
        return log_p


class TruncatedStandardNormal(Distribution):
    """
    Represents a truncated standard normal distribution which is the standard normal distribution
    constrained between two bounds a and b. It modifies the standard normal by zeroing out the probability
    density outside the interval [a, b] and renormalizing the probability density inside [a, b].

    Attributes:
        a (Tensor): Lower bound of the truncation.
        b (Tensor): Upper bound of the truncation.

    Args:
        a (float or Tensor): The lower bound of the truncation.
        b (float or Tensor): The upper bound of the truncation.
        validate_args (bool, optional): Whether to validate input arguments.

    Formulas:
        - Probability Density Function (PDF):
            f(x | a, b) = φ(x) / (Φ(b) - Φ(a))
              where φ(x) is the PDF of the standard normal distribution,
              and Φ(x) is the CDF of the standard normal distribution.

        - Cumulative Distribution Function (CDF):
            F(x | a, b) = (Φ(x) - Φ(a)) / (Φ(b) - Φ(a))

        - Mean:
            E[X | a, b] = -(φ(b) - φ(a)) / (Φ(b) - Φ(a))

        - Variance:
            Var(X | a, b) = 1 + (aφ(a) - bφ(b)) / (Φ(b) - Φ(a)) - (E[X | a, b])^2

        - Entropy:
            H(X | a, b) = constant + log(Φ(b) - Φ(a)) - 0.5 * (aφ(a) - bφ(b)) / (Φ(b) - Φ(a))        

    References:
        https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """
    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        
        # Precalculate constants for numerical stability
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        
        # Compute the density and distribution functions at the bounds
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)

        # Normalizing constant
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        
        # Precompute terms needed for mean and variance
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        
        # Compute mean and variance of the truncated distribution
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        
        # Compute entropy estimation
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def sample(self, sample_shape=None):
        sample_shape = [sample_shape] if isinstance(sample_shape, Number) else sample_shape
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

    def __call__(self, num_samples):
        z = self.sample(num_samples)
        lp = self.log_prob(z)
        return z, lp

class TruncatedNormal(TruncatedStandardNormal):
    """
    Represents a truncated normal distribution, which is a general normal distribution
    constrained between two bounds a and b, unlike TruncatedStandardNormal that specifically
    deals with a standard normal distribution (mean = 0, variance = 1) truncated between two bounds.

    This class extends TruncatedStandardNormal by incorporating a location parameter `loc` and a
    scale parameter `scale`, transforming the distribution to have mean `loc` and variance `scale^2`.

    Attributes:
        loc (Tensor or Parameter): The location (mean) parameter of the normal distribution.
        scale (Tensor or Parameter): The scale (standard deviation) parameter of the normal distribution.
        a (Tensor): Lower bound of the truncation after adjusting for `loc` and `scale`.
        b (Tensor): Upper bound of the truncation after adjusting for `loc` and `scale`.
        dim (int): Dimensionality of the distribution parameters.
        device (str): The device on which the tensors are stored.
    
    Args:
        dim (int): Dimensionality of the distribution parameters.
        loc (float or Tensor): The location (mean) parameter of the normal distribution.
        scale (float or Tensor): The scale (standard deviation) parameter of the normal distribution.
        a (float or Tensor): The initial lower bound of the truncation (before transformation).
        b (float or Tensor): The initial upper bound of the truncation (before transformation).
        trainable (bool, optional): If True, the location and scale parameters are trainable (using nn.Parameter).
        device (str, optional): The device on which to store the tensors.
        validate_args (bool, optional): Whether to validate input arguments.
    """
    
    has_rsample = True

    def __init__(self,dim, loc, scale, a, b,trainable = False,device='cuda', validate_args=None):

        # Store initial truncation bounds, dimensionality, and device information
        self.a = a
        self.b = b        
        self.dim = dim
        self.device = device

        # Initialize location and scale parameters, possibly as trainable parameters
        if trainable:
            self.loc = nn.Parameter(torch.zeros((dim,), device=self.device)) + loc
            self.scale = nn.Parameter(torch.ones(self.dim, device=self.device)) * scale
        else:
            self.loc = torch.zeros((dim,), device=self.device) + loc
            self.scale = torch.ones(self.dim, device=self.device) * scale

        a = torch.zeros((dim,), device=self.device) + a
        b = torch.zeros((dim,), device=self.device) + b

        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)

        # Normalize a and b to create a standard truncated normal
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale

        # Initialize the base class with the normalized bounds
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)

        # Adjust the distribution properties for the non-standardized form -- checked for num instability, couldn't find any, need to find the 1986 Pierre something paper though!
        self._log_scale = self.scale.log()  # Log of the scale parameter
        self._mean = self._mean * self.scale + self.loc  # Adjust mean for the non-standardized form
        self._variance = self._variance * self.scale ** 2  # Adjust variance for the non-standardized form
        self._entropy += self._log_scale  # Adjust entropy for the scale


    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        _lp = super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale
        return _lp.sum(1)




