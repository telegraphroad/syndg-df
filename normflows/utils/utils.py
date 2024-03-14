import torch
import torch.nn as nn

def tukey_biweight_estimator(tensor, initial_guess=None, c=1.345, max_iter=100, tol=1e-6):
    """
    Compute the Tukey biweight M-estimator for a 1D tensor.
    This function estimates the central location of the data robustly, minimizing the influence of outliers.

    Parameters:
    - tensor: 1D tensor, the input data
    - initial_guess: float or None, optional initial guess for the estimator
    - c: float, tuning constant that determines the cutoff for what is considered an outlier (default 1.345)
    - max_iter: int, maximum number of iterations to refine the estimator (default 100)
    - tol: float, convergence tolerance, the loop stops if changes are smaller than this value (default 1e-6)

    Returns:
    - mu: float, the estimated central location of the data using the Tukey biweight method
    """
    # Use the median as a starting point if no initial guess is provided
    if initial_guess is None:
        mu = tensor.median()
    else:
        mu = initial_guess

    # Iterate to refine the estimate of mu
    for _ in range(max_iter):
        # Calculate differences from the current estimate mu
        diffs = tensor - mu

        # Calculate weights: points within [-c, c] have weight 1, others are down-weighted
        weights = torch.where(torch.abs(diffs) <= c, torch.ones_like(tensor), c / torch.abs(diffs))

        # Update the estimate of mu weighted by the calculated weights
        mu_next = torch.sum(weights * tensor) / torch.sum(weights)

        # Check for convergence: if the change is smaller than the tolerance, stop iterating
        if torch.abs(mu - mu_next) < tol:
            break

        # Update mu for the next iteration
        mu = mu_next

    return mu


def geometric_median_of_means_pyt(samples, num_buckets, max_iter=100, eps=1e-5):
    """
    Compute the geometric median of means by partitioning `samples` into `num_buckets`.
    The geometric median is robust to outliers and is computed using Weiszfeld's algorithm.

    Parameters:
    - samples: tensor, input data samples
    - num_ways: int, number of buckets to divide the samples into
    - max_iter: int, maximum number of iterations for the iterative algorithm (default 100)
    - eps: float, convergence criterion threshold, algorithm stops if relative change is below this value (default 1e-5)

    Returns:
    - gmom_est: tensor, the estimated geometric median of the bucketed means
    """
    # Ensure samples are at least 2D
    if len(samples.shape) == 1:
        samples = samples.reshape(-1, 1)
    
    # Compute the mean of each bucket
    bucketed_means = torch.stack(
        [torch.mean(val, dim=0) for val in torch.split(samples, num_buckets)]
    )

    # Directly return the mean if there's only one bucket
    if bucketed_means.shape[0] == 1:
        return bucketed_means.squeeze()

    # Initialize the geometric median estimate as the mean of bucketed means
    gmom_est = torch.mean(bucketed_means, dim=0)

    # Iterate to refine the geometric median estimate
    for i in range(max_iter):
        # Compute weights inversely proportional to the Euclidean distance from current estimate
        weights = 1 / torch.norm(bucketed_means - gmom_est, dim=1, p=2)[:, None]
        
        # Save the current estimate to check for convergence
        old_gmom_est = gmom_est
        
        # Update the estimate using the weighted average of bucketed means
        gmom_est = (bucketed_means * weights).sum(dim=0) / weights.sum()
        
        # Check for convergence: stop if relative change is below the threshold
        if (
            torch.norm(gmom_est - old_gmom_est, p=2) / torch.norm(old_gmom_est, p=2)
            < eps
        ):
            break

    return gmom_est