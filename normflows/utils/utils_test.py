import unittest
import torch
from .utils import geometric_median_of_means_pyt, tukey_biweight_estimator  # Ensure this imports correctly

class TestGeometricMedianOfMeans(unittest.TestCase):
    
    def test_single_value(self):
        """ Test with a single value tensor """
        samples = torch.tensor([1.0])
        result = geometric_median_of_means_pyt(samples, num_buckets=1)
        self.assertTrue(torch.isclose(result, torch.tensor(1.0)))

    def test_single_bucket(self):
        """ Test with all samples in a single bucket """
        samples = torch.tensor([1.0, 2.0, 3.0])
        result = geometric_median_of_means_pyt(samples, num_buckets=1)
        self.assertTrue(torch.isclose(result, torch.mean(samples)))

    def test_multiple_buckets(self):
        """ Test with multiple buckets, each having a single element """
        samples = torch.tensor([1.0, 2.0, 3.0])
        result = geometric_median_of_means_pyt(samples, num_buckets=3)
        self.assertTrue(torch.isclose(result, torch.tensor(2.0)))

    def test_high_dimensionality(self):
        """ Test with high-dimensional input samples """
        samples = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        result = geometric_median_of_means_pyt(samples, num_buckets=3)
        expected_median = torch.tensor([2.0, 3.0])
        self.assertTrue(torch.allclose(result, expected_median))

    def test_convergence(self):
        """ Test to ensure convergence within a reasonable number of iterations """
        samples = torch.rand(100) * 100  # Random data
        result = geometric_median_of_means_pyt(samples, num_buckets=10, max_iter=1000)
        # No specific value to check against, just ensuring it runs without error
        self.assertIsInstance(result, torch.Tensor)

    def test_non_divisible_buckets(self):
        """ Test with a number of samples not divisible by num_buckets """
        samples = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = geometric_median_of_means_pyt(samples, num_buckets=2)
        # Exact median cannot be determined but checking it completes
        self.assertIsInstance(result, torch.Tensor)

    def test_zero_buckets_error(self):
        """ Test to handle edge case where num_buckets is zero, expecting an error """
        samples = torch.tensor([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            geometric_median_of_means_pyt(samples, num_buckets=0)

    def test_negative_buckets_error(self):
        """ Test to handle negative buckets, expecting an error """
        samples = torch.tensor([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            geometric_median_of_means_pyt(samples, num_buckets=-1)

class TestTukeyBiweightEstimator(unittest.TestCase):

    def test_basic_functionality(self):
        """Test with normal input."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])  # Including an outlier
        result = tukey_biweight_estimator(tensor)
        self.assertTrue(torch.isclose(result, torch.tensor(2.5), atol=1e-4))

    def test_with_initial_guess(self):
        """Test the function's response to an initial guess."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
        initial_guess = 2.0
        result = tukey_biweight_estimator(tensor, initial_guess)
        self.assertTrue(torch.isclose(result, torch.tensor(2.5), atol=1e-4))

    def test_high_c_value(self):
        """Test with a high c value, which should consider all points equally."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
        result = tukey_biweight_estimator(tensor, c=1000)  # Large c value
        expected = torch.sum(tensor) / len(tensor)  # Should be close to simple mean
        self.assertTrue(torch.isclose(result, expected, atol=1e-4))

    def test_zero_iterations(self):
        """Test with zero iterations to check behavior with initial guess only."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
        initial_guess = 2.0
        result = tukey_biweight_estimator(tensor, initial_guess=initial_guess, max_iter=0)
        self.assertEqual(result, initial_guess)

    def test_convergence_tolerance(self):
        """Test the function's ability to stop based on the tolerance."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # Small changes in mu should lead to early stopping
        result = tukey_biweight_estimator(tensor, tol=0.5)
        self.assertTrue(isinstance(result, float))

    def test_empty_tensor(self):
        """Test how the function handles an empty tensor."""
        tensor = torch.tensor([])
        with self.assertRaises(ValueError):
            tukey_biweight_estimator(tensor)

    def test_singular_value_tensor(self):
        """Test a tensor with all identical values."""
        tensor = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
        result = tukey_biweight_estimator(tensor)
        self.assertEqual(result, 5.0)

if __name__ == '__main__':
    unittest.main()