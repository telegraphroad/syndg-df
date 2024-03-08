import os
import unittest
from io import StringIO

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import gamma, multivariate_normal
from scipy.stats import multivariate_t, t
from scipy.stats import truncnorm
from torch.distributions import Categorical, MixtureSameFamily
from ..utils.preprocessing import CSVDataset
from .base_extended import (GaussianMixture, GeneralizedGaussianDistribution,TruncatedStandardNormal,
                            GeneralizedGaussianMixture, StudentTDistribution, MultivariateStudentTDist, StudentTMixture)


class TestGeneralizedGaussianDistribution(unittest.TestCase):
    def setUp(self):
        # Parameters for the distribution
        self.loc = torch.tensor(0.0)
        self.scale = torch.tensor(1.0)
        self.p = torch.tensor(1.5)  # Corresponds to a normal distribution

        # Create instance of the GeneralizedGaussianDistribution
        self.dist = GeneralizedGaussianDistribution(self.loc, self.scale, self.p)

    def test_initialization(self):
        # Check if the parameters are set correctly
        self.assertEqual(self.dist.loc, self.loc)
        self.assertEqual(self.dist.scale, self.scale)
        self.assertEqual(self.dist.p, self.p)

    def test_mean(self):
        # Mean should be equal to the location parameter
        self.assertEqual(self.dist.mean, self.loc)

    def test_variance(self):
        # Compare variance calculation with scipy's implementation
        expected_variance = self.scale.pow(2) * (torch.exp(torch.lgamma(torch.tensor(3.0)/self.p) - torch.lgamma(torch.tensor(1.0)/self.p)))
        self.assertAlmostEqual(self.dist.variance.item(), expected_variance.item(), places=5)

    def test_sample(self):
        # Check the shape of the sample
        sample = self.dist.sample(torch.Size([10]))
        self.assertEqual(sample.shape, torch.Size([10]))

    def test_log_prob(self):
        # Evaluate log probability and compare with scipy's logpdf
        value = torch.tensor(0.5)
        expected_log_prob = stats.gennorm.logpdf(value, beta=self.p, loc=self.loc, scale=self.scale)
        calculated_log_prob = self.dist.log_prob(value)
        self.assertAlmostEqual(calculated_log_prob.item(), expected_log_prob, places=5)

    def test_cdf(self):
        # Evaluate CDF and compare with scipy's cdf
        value = torch.tensor(0.5)
        expected_cdf = stats.gennorm.cdf(value, beta=self.p, loc=self.loc, scale=self.scale)
        calculated_cdf = self.dist.cdf(value)
        self.assertAlmostEqual(calculated_cdf.item(), expected_cdf, places=5)

    def test_entropy(self):
        # Check the entropy calculation
        expected_entropy = (1/self.p) - torch.log(self.p) + torch.log(2*self.scale) + torch.lgamma(1/self.p)
        self.assertAlmostEqual(self.dist.entropy().item(), expected_entropy.item(), places=5)

    def test_not_implemented_methods(self):
        # Check if NotImplementedError is raised for icdf
        with self.assertRaises(NotImplementedError):
            self.dist.icdf(torch.tensor(0.5))

class TestGeneralizedGaussianMixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary CSV file
        cls.temp_csv_path = 'temp_test_data.csv'
        pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        }).to_csv(cls.temp_csv_path, index=False)

    def test_initialization(self):
        dataset = CSVDataset(self.temp_csv_path)
        ggm = GeneralizedGaussianMixture(n_modes=2, dim=2, ds=dataset)
        self.assertEqual(ggm.n_modes, 2)
        self.assertEqual(ggm.dim, 2)
        self.assertTrue(torch.allclose(ggm.loc, torch.from_numpy(np.tile(dataset.calculate_feature_means(), (2, 1)))))

    def test_log_prob(self):
        dataset = CSVDataset(self.temp_csv_path)
        ggm = GeneralizedGaussianMixture(n_modes=1, dim=1, ds=dataset, p=2, scale=1, loc=0)
        x = torch.tensor([0.5], dtype=torch.float)
        log_prob_scipy = gamma.logpdf(np.abs(x.numpy() - ggm.loc.numpy()), a=1/ggm.p.numpy(), scale=ggm.scale.numpy())
        log_prob_torch = ggm.log_prob(x)
        np.testing.assert_almost_equal(log_prob_torch.item(), log_prob_scipy.sum(), decimal=5)

    @classmethod
    def tearDownClass(cls):
        import os
        os.remove(cls.temp_csv_path)

class TestGaussianMixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary CSV file
        cls.temp_csv_path = 'temp_test_data.csv'
        pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        }).to_csv(cls.temp_csv_path, index=False)

    def test_initialization(self):
        dataset = CSVDataset(self.temp_csv_path)
        gm = GaussianMixture(n_modes=2, dim=2, ds=dataset)
        self.assertEqual(gm.n_modes, 2)
        self.assertEqual(gm.dim, 2)
        self.assertTrue(isinstance(gm.mixture, MixtureSameFamily))

    def test_log_prob(self):
        dataset = CSVDataset(self.temp_csv_path)
        gm = GaussianMixture(n_modes=1, dim=1, ds=dataset, loc=0, scale=1, p=2, rand_p=False)
        x = torch.tensor([0.5], dtype=torch.float).to(gm.device)
        scipy_log_prob = multivariate_normal(mean=gm.loc.cpu().detach().numpy(), cov=gm.scale.cpu().detach().numpy()**2).logpdf(x.cpu().numpy())
        torch_log_prob = gm.log_prob(x)
        np.testing.assert_almost_equal(torch_log_prob.item(), scipy_log_prob, decimal=5)

    @classmethod
    def tearDownClass(cls):
        import os
        os.remove(cls.temp_csv_path)

class TestStudentTDistribution(unittest.TestCase):

    def test_initialization(self):
        # Test with different shapes and device settings
        dist1 = StudentTDistribution(shape=(2, 3), device='cpu')
        self.assertEqual(dist1.n_dim, 2)
        self.assertTrue(isinstance(dist1.df, torch.nn.Parameter))
        self.assertTrue(dist1.df.item() == 2.0)
        self.assertEqual(dist1.loc.shape, (1, 2, 3))
        self.assertEqual(dist1.log_scale.shape, (1, 2, 3))

        # Test with non-trainable parameters
        dist2 = StudentTDistribution(shape=5, trainable=False, device='cpu')
        self.assertTrue(isinstance(dist2.df, torch.Tensor))
        self.assertFalse(dist2.df.requires_grad)
        self.assertEqual(dist2.loc.shape, (1, 5))
        self.assertFalse(dist2.loc.requires_grad)

    def test_log_prob(self):
        # Initialize the distribution with specific parameters
        dist = StudentTDistribution(shape=(1,), df=3.0, device='cpu')
        sample = torch.tensor([0.5], dtype=torch.float)
        
        # Calculate log probability using the defined class
        log_prob_torch = dist.log_prob(sample)

        # Calculate expected log probability using scipy
        log_prob_scipy = t.logpdf(sample.numpy(), df=3.0, loc=0, scale=1)

        # Compare both results
        np.testing.assert_almost_equal(log_prob_torch.item(), log_prob_scipy, decimal=5)

class TestMultivariateStudentTDist(unittest.TestCase):

    def test_initialization(self):
        # Test with trainable parameters
        mvt = MultivariateStudentTDist(degree_of_freedom=3, dim=2, trainable=True, device='cpu')
        self.assertEqual(mvt.dim, 2)
        self.assertTrue(isinstance(mvt.df, torch.nn.Parameter))
        self.assertTrue(mvt.df.item() == 3)
        self.assertTrue(torch.equal(mvt.loc, torch.zeros(2)))
        self.assertTrue(torch.equal(mvt.scale_tril, torch.eye(2)))

        # Test with non-trainable parameters
        mvt_non_trainable = MultivariateStudentTDist(degree_of_freedom=4, dim=3, trainable=False, device='cpu')
        self.assertFalse(mvt_non_trainable.df.requires_grad)
        self.assertTrue(torch.equal(mvt_non_trainable.loc, torch.zeros(3)))
        self.assertTrue(torch.equal(mvt_non_trainable.scale_tril, torch.eye(3)))

    def test_log_prob(self):
        # Initialize the distribution with specific parameters
        mvt = MultivariateStudentTDist(degree_of_freedom=5, dim=2, trainable=False, device='cpu')
        samples = torch.tensor([[0.1, -0.1], [0.2, 0.2]], dtype=torch.float)
        
        # Calculate log probability using the defined class
        log_prob_torch = mvt.log_prob(samples)

        # Calculate expected log probability using scipy
        df, loc, scale = 5, [0, 0], np.eye(2)
        log_prob_scipy = multivariate_t.logpdf(samples.numpy(), df, loc, scale)

        # Compare both results
        np.testing.assert_array_almost_equal(log_prob_torch.numpy(), log_prob_scipy, decimal=4)

class TestStudentTMixture(unittest.TestCase):

    def test_initialization(self):
        # Initialize with specific parameters
        n_modes, dim = 3, 2
        mixture = StudentTMixture(n_modes=n_modes, dim=dim, loc=0., scale=1., p=2., rand_p=False, device='cpu')

        # Check attributes
        self.assertEqual(mixture.n_modes, n_modes)
        self.assertEqual(mixture.dim, dim)
        self.assertTrue(isinstance(mixture.loc, torch.Tensor))
        self.assertTrue(isinstance(mixture.scale, torch.Tensor))
        self.assertTrue(isinstance(mixture.p, torch.Tensor))
        self.assertTrue(isinstance(mixture.weight_scores, torch.Tensor))

    def test_log_prob(self):
        # Create a simple two-mode mixture for easy calculation
        n_modes, dim = 2, 1
        loc = np.array([[0], [1]])
        scale = np.array([[1], [1]])
        p = np.array([[2], [2]])
        weights = np.array([0.5, 0.5])

        mixture = StudentTMixture(n_modes=n_modes, dim=dim, loc=loc, scale=scale, p=p, weights=weights, rand_p=False, device='cpu')

        # Sample from the mixture
        samples, _ = mixture.forward(num_samples=100)

        # Calculate log probability using the class
        log_prob_torch = mixture.log_prob(samples)

        # Calculate expected log probability manually
        log_probs = []
        for i in range(n_modes):
            df = p[i, 0] * 2
            rv = multivariate_t(loc=loc[i], shape=scale[i], df=df)
            log_prob_i = rv.logpdf(samples.numpy()) + np.log(weights[i])
            log_probs.append(log_prob_i)
        log_prob_expected = np.logaddexp(log_probs[0], log_probs[1])

        # Compare both results
        np.testing.assert_array_almost_equal(log_prob_torch.numpy(), log_prob_expected, decimal=4)

class TestTruncatedStandardNormal(unittest.TestCase):
    
    def setUp(self):
        # Constants for the tests
        self.a = -1.0
        self.b = 1.0
        self.distribution = TruncatedStandardNormal(a=self.a, b=self.b)

    def test_initialization(self):
        # Check initial properties like mean, variance, etc.
        self.assertAlmostEqual(self.distribution.mean.item(), 0, places=3)
        self.assertTrue(self.distribution.variance.item() < 1)  # Variance should be less than non-truncated std normal

    def test_bounds(self):
        # Ensure a < b and they are set correctly
        self.assertLess(self.distribution.a, self.distribution.b)

    def test_log_prob(self):
        # Test log probabilities at specific points
        test_values = torch.tensor([-0.5, 0, 0.5])
        actual_log_probs = self.distribution.log_prob(test_values)
        expected_log_probs = truncnorm.logpdf(test_values.numpy(), (self.a - 0) / 1, (self.b - 0) / 1, loc=0, scale=1)
        
        np.testing.assert_array_almost_equal(actual_log_probs.numpy(), expected_log_probs, decimal=4)

    def test_cdf_icdf(self):
        # Test CDF and inverse CDF
        test_values = torch.tensor([-0.5, 0, 0.5])
        cdf_values = self.distribution.cdf(test_values)
        icdf_values = self.distribution.icdf(cdf_values)
        
        np.testing.assert_array_almost_equal(icdf_values.numpy(), test_values.numpy(), decimal=4)

    def test_sample(self):
        # Test sampling
        samples = self.distribution.sample((1000,))
        self.assertTrue(torch.all(samples >= self.a) and torch.all(samples <= self.b))

    def test_entropy(self):
        # Test computed entropy
        expected_entropy = truncnorm.entropy((self.a - 0) / 1, (self.b - 0) / 1, loc=0, scale=1)
        self.assertAlmostEqual(self.distribution.entropy.item(), expected_entropy, places=4)


if __name__ == '__main__':
    unittest.main()