import torch
import math


class GaussianWeight(object):
    """A Gaussian Variational Distribution for NN weights
    """
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.sigma = torch.log1p(torch.exp(self.rho))
        
    def sample(self):
        """Draw sample from the distribution
        """
        epsilon = torch.distributions.Normal(0,1).sample(self.rho.size())
        sample = self.mu + self.sigma*epsilon
        return sample 
    
    def log_prob(self, x):
        """Log PDF
        """
        return (-math.log(math.sqrt(2*math.pi)) - torch.log(self.sigma) - ( ((x - self.mu)**2)/(2 * self.sigma**2) )).sum()
    


class ScaledMixedGaussian(object):
    """Sacaled Mixture Gaussian Prior for NN weights
    """
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.Gaussian1 = torch.distributions.Normal(0, self.sigma1)
        self.Gaussian2 = torch.distributions.Normal(0, self.sigma2)
    
    def log_prob(self, x):
        """Log PDF
        """
        pd1 = torch.exp(self.Gaussian1.log_prob(x))
        pd2 = torch.exp(self.Gaussian2.log_prob(x))
        return (torch.log(self.pi*pd1 + (1-self.pi)*pd2)).sum()
        