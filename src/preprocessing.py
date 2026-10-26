"""
    This module includes the preprocessing functions for ERA5 and Satellite data, including data loading, filtering, and normalization.
"""

import os
import sys
from datetime import datetime

import numpy as np
import netCDF4 as nc

from sklearn.decomposition import PCA

def as_float_array(values: np.ndarray) -> np.ndarray:
    """
    Convert the input array to a float array, masking any invalid values 
    (e.g., NaN or Inf) and filling them with NaN.
    """

    masked_values = np.ma.asarray(values, dtype=np.float64)
    return np.asarray(masked_values.filled(np.nan))

def calculate_eof(data: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the EOF/PC calculation used in the original notebook."""

    # creating PCA object
    pca: PCA = PCA(n_components=n_components)

    # fit PCA to the data
    pca.fit(data.T)
    eofs = pca.components_.T
    pcs = pca.transform(data.T).T

    return eofs, pcs

def convert_time(time_variable: nc.Variable) -> np.ndarray:
    """Read daily dates from CF time coordinates or the ERA5 YYYYMMDD format."""
    if time_variable.units == "day as %Y%m%d.%f":
        dates = [
            datetime.strptime(str(int(value)), "%Y%m%d")
            for value in time_variable[:]
        ]
    else:
        dates = nc.num2date(
            time_variable[:],
            units=time_variable.units,
            calendar=getattr(time_variable, "calendar", "standard"),
        )
    return np.asarray([date.strftime("%Y-%m-%d") for date in dates]) #type: ignore