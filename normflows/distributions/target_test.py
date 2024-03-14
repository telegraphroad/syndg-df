import unittest

import torch
from torch.distributions import Normal

from normflows.distributions.base import DiagGaussian
from normflows.distributions.target import TwoMoons, \
    CircularGaussianMixture, RingMixture, TwoIndependent, \
    ConditionalDiagGaussian

from .target_extended import NealsFunnel


class TargetTest(unittest.TestCase):
    def test_targets(self):
        targets = [TwoMoons, CircularGaussianMixture,
                   RingMixture]
        for num_samples in [1, 5]:
            for target_ in targets:
                with self.subTest(num_samples=num_samples,
                                  target_=target_):
                    # Set up target
                    target = target_()
                    # Test target
                    samples = target.sample(num_samples)
                    assert samples.shape == (num_samples, 2)
                    log_p = target.log_prob(samples)
                    assert log_p.shape == (num_samples,)

    def test_two_independent(self):
        target = TwoIndependent(TwoMoons(), DiagGaussian(2))
        for num_samples in [1, 5]:
            with self.subTest(num_samples=num_samples):
                # Test target
                samples = target.sample(num_samples)
                assert samples.shape == (num_samples, 4)
                log_p = target.log_prob(samples)
                assert log_p.shape == (num_samples,)

    def test_conditional(self):
        target = ConditionalDiagGaussian()
        for num_samples in [1, 5]:
            for size in [1, 3]:
                with self.subTest(num_samples=num_samples,
                                  size=size):
                    context = torch.rand(num_samples, size * 2) + 0.5
                    # Test target
                    samples = target.sample(num_samples, context)
                    assert samples.shape == (num_samples, size)
                    log_p = target.log_prob(samples, context)
                    assert log_p.shape == (num_samples,)

class TestNealsFunnel(unittest.TestCase):
    def test_default_initialization(self):
        funnel = NealsFunnel()
        self.assertEqual(funnel.prop_scale.item(), 20)
        self.assertEqual(funnel.prop_shift.item(), -10)
        self.assertEqual(funnel.v1shift, 0)
        self.assertEqual(funnel.v2shift, 0)

    def test_parameterized_initialization(self):
        funnel = NealsFunnel(torch.tensor(10.), torch.tensor(-5.), 1., 2.)
        self.assertEqual(funnel.prop_scale.item(), 10)
        self.assertEqual(funnel.prop_shift.item(), -5)
        self.assertEqual(funnel.v1shift, 1)
        self.assertEqual(funnel.v2shift, 2)

    def test_log_prob_correctness(self):
        funnel = NealsFunnel()
        z = torch.tensor([[0., 0.], [1., 1.]])
        expected_v_like = Normal(torch.tensor([0.0]), torch.tensor([1.0])).log_prob(z[:, 0])
        expected_x_like = Normal(torch.tensor([0.0]), torch.exp(0.5 * z[:, 0])).log_prob(z[:, 1])
        expected_log_prob = expected_v_like + expected_x_like
        computed_log_prob = funnel.log_prob(z)
        self.assertTrue(torch.allclose(computed_log_prob, expected_log_prob))

    def test_edge_cases(self):
        funnel = NealsFunnel()
        z_empty = torch.tensor([])
        with self.assertRaises(IndexError):
            funnel.log_prob(z_empty)

        z_inf = torch.tensor([[float('inf'), 0.], [0., float('inf')]])
        log_prob_inf = funnel.log_prob(z_inf)
        self.assertTrue(torch.isinf(log_prob_inf).any())

if __name__ == "__main__":
    unittest.main()