"""
    This module includes the preprocessing functions for ERA5 and Satellite data, including data loading, filtering, and normalization.
"""

import os
import sys

import numpy as np

def as_float_array(values: np.ndarray) -> np.ndarray:
    """
    Convert the input array to a float array, masking any invalid values 
    (e.g., NaN or Inf) and filling them with NaN.
    """

    masked_values = np.ma.asarray(values, dtype=np.float64)
    return np.asarray(masked_values.filled(np.nan))

def calculate_eof(data: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the EOF/PC calculation used in the original notebook."""
    _, n_samples = data.shape
    anomalies = data - np.nanmean(data, axis=1, keepdims=True)
    covariance = (anomalies @ anomalies.T) / n_samples
    _, eigenvectors = np.linalg.eig(covariance)
    eofs = eigenvectors[:, :n_components]
    eofs = eofs / np.sum(eofs, axis=0)
    pcs = np.linalg.inv(eofs.T @ eofs) @ (eofs.T @ anomalies)
    return eofs, pcs