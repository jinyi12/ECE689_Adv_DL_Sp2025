import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
import math
class MutualInformationEstimator(ABC):
    """
    Base class for mutual information estimation.

    This class provides a common interface for mutual information (MI) estimators,
    where MI can be estimated directly (e.g. using histogram binning) or indirectly using
    kernel-based dependence measures (e.g. HSIC via random Fourier features).

    It assumes the input is a torch.Tensor of shape (n_samples, n_variables).
    Note: Some estimators (e.g., RFFMI below) may return dependence measures that are not directly
          convertible to mutual information.
    """
    def __init__(self, samples: torch.Tensor):
        self.samples = samples
        self.n_samples = samples.shape[0]
        self.n_variables = samples.shape[1]
        
    @abstractmethod
    def compute_pairwise_mi(self) -> torch.Tensor:
        """Compute mutual information or a dependence measure between all pairs of variables."""
        pass

class BinningMI(MutualInformationEstimator):
    """Mutual information estimation using binning approach"""
    def __init__(self, samples: torch.Tensor, bins: Optional[int] = None):
        super().__init__(samples)
        self.bins = bins
        
    def _estimate_optimal_bins(self, x: torch.Tensor) -> int:
        """Estimate optimal number of bins using Freedman-Diaconis rule
        
        FD rule: bin width = 2 * IQR * n^(-1/3)
        where IQR is the interquartile range
        """
        x_np = x.numpy()
        q75, q25 = np.percentile(x_np, [75, 25])
        iqr = q75 - q25
        n = len(x_np)
        
        # Handle edge cases
        if iqr == 0:
            return 20  # fallback for constant/low-variance data
            
        # Compute bin width using FD rule
        bin_width = 2 * iqr * (n ** (-1/3))
        data_range = np.ptp(x_np)  # peak-to-peak (max - min)
        
        n_bins = int(np.ceil(data_range / bin_width))
        return max(10, min(n_bins, 50))  # Keep bins between 10 and 50
    
    def _compute_mi_pair(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute MI between two variables using binning"""
        if self.bins is None:
            n_bins = min(self._estimate_optimal_bins(x), self._estimate_optimal_bins(y))
        else:
            n_bins = self.bins
            
        # Compute 2D histogram
        hist_xy, _, _ = np.histogram2d(
            x.numpy(), y.numpy(), bins=n_bins,
            density=True
        )
        
        # Add smoothing to avoid log(0)
        hist_xy += 1e-10
        
        # Normalize to get probabilities
        p_xy = hist_xy / hist_xy.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        # Compute MI
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i,j] > 0:
                    mi += p_xy[i,j] * np.log(p_xy[i,j] / (p_x[i] * p_y[j]))
        return float(mi)
    
    def compute_pairwise_mi(self) -> torch.Tensor:
        """Compute MI between all pairs of variables"""
        mi_matrix = torch.zeros((self.n_variables, self.n_variables))
        
        for i in range(self.n_variables):
            for j in range(i + 1, self.n_variables):
                mi = self._compute_mi_pair(
                    self.samples[:, i],
                    self.samples[:, j]
                )
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
                
        return mi_matrix

class RFFMI(MutualInformationEstimator):
    """
    Mutual Information estimation using Random Fourier Features (RFF) for kernel-based dependence measurement.
    
    This estimator computes a kernel-based dependence measure using the Hilbert-Schmidt Independence
    Criterion (HSIC) between variables. The gamma parameter is determined using a median heuristic:
        gamma = 1.0 / (2 * median(pairwise squared distances)).
    
    Random Fourier features are used to approximate a Gaussian kernel (with variance 1/(2*gamma)).
    
    Note:
      - The returned value from compute_pairwise_mi is the squared Frobenius norm of the cross-covariance matrix
        between the RFF projections of pairs of variables. It does not have a direct conversion to mutual information.
      - For univariate inputs, an extra dimension is automatically added.
    """
    def __init__(self, samples: torch.Tensor, n_features: int = 100):
        super().__init__(samples)
        self.n_features = n_features
        self.gamma = self._median_heuristic()

    def _median_heuristic(self):
        """Compute gamma using the median heuristic.
        
        It computes the squared pairwise distances between samples and sets:
            gamma = 1.0 / (2 * median(pairwise squared distances))
        """
        pairwise_dists = torch.cdist(self.samples, self.samples)**2
        return 1.0 / (2 * torch.median(pairwise_dists))

    def _compute_rff(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is 2D; if univariate, add an extra dimension
        if x.dim() == 1:
            x = x.unsqueeze(1)
        w = torch.randn(x.shape[1], self.n_features, device=x.device)
        w *= math.sqrt(2 * self.gamma)
        b = torch.rand(self.n_features, device=x.device) * 2 * math.pi
        return torch.cos(x @ w + b) * math.sqrt(2.0/self.n_features)

    def _compute_kmi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute kernel mutual information using Random Fourier Features (RFF).
        
        This method uses RFF to approximate the Gaussian kernel for x and y, computes the covariance matrices,
        and then performs a whitening procedure to obtain canonical correlations. Assuming a Gaussian model in
        the RFF feature space, the mutual information is estimated as:
            MI = -0.5 * sum(log(1 - rho^2))
        where rho are the canonical correlations.
        """
        # Compute random Fourier features for x and y.
        z_x = self._compute_rff(x)
        z_y = self._compute_rff(y)
        n = self.n_samples
        device = self.samples.device
        eps = 1e-6
        I = torch.eye(self.n_features, device=device)

        # Estimate covariance matrices; add regularization for numerical stability.
        C_x = (z_x.T @ z_x) / n + eps * I
        C_y = (z_y.T @ z_y) / n + eps * I
        C_xy = (z_x.T @ z_y) / n

        # Perform Cholesky decompositions.
        L_x = torch.linalg.cholesky(C_x)
        L_y = torch.linalg.cholesky(C_y)
        inv_L_x = torch.linalg.inv(L_x)
        inv_L_y = torch.linalg.inv(L_y)

        # Whiten the cross-covariance.
        M = inv_L_x.T @ C_xy @ inv_L_y

        # Use SVD to get canonical correlations.
        U, s, V = torch.linalg.svd(M)
        s = torch.clamp(s, max=0.9999)
        mi_value = -0.5 * torch.sum(torch.log(1 - s**2 + 1e-10))
        return mi_value.item()

    def compute_pairwise_mi(self) -> torch.Tensor:
        """Compute pairwise kernel mutual information between all variable pairs using random Fourier features.
        
        Returns:
            A symmetric matrix of shape (n_variables, n_variables) where the (i, j)-th entry
            is the estimated mutual information computed from the canonical correlations of the RFF projections
            for the i-th and j-th variables.
        
        Note:
            The estimator assumes a Gaussian model in the RFF-induced feature space and computes:
                MI = -0.5 * sum(log(1 - rho^2))
            where rho are the canonical correlations.
        """
        mi_matrix = torch.zeros((self.n_variables, self.n_variables), device=self.samples.device)
        
        for i in range(self.n_variables):
            for j in range(i + 1, self.n_variables):
                # Extract individual variable data.
                x_i = self.samples[:, i]
                x_j = self.samples[:, j]
                # Ensure the variables are 2D (for univariate data, add an extra dimension)
                if x_i.dim() == 1:
                    x_i = x_i.unsqueeze(1)
                if x_j.dim() == 1:
                    x_j = x_j.unsqueeze(1)
                
                mi_val = self._compute_kmi(x_i, x_j)
                mi_matrix[i, j] = mi_val
                mi_matrix[j, i] = mi_val
                
        return mi_matrix