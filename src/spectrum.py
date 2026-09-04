# ====================================================
# This script is to do cross-spectral analysis
# ====================================================

# ====================================================
# Import package
# ====================================================

import numpy as np

from scipy.signal import detrend
from typing import Tuple

# ====================================================
# Pre-processing functions
# ====================================================

# symmetrizing / asymmetrizing data
def symm_asym(
        data   : np.ndarray,
        lat_axis: int
) -> Tuple[np.ndarray, ...]:

    """
    Symmetrizing / Asymetrizing data along meridional direction
    following Wheeler and Kiladis (1999).
    """

    symm_data: np.ndarray = (data + np.flip(data, axis=lat_axis)) / 2
    asym_data: np.ndarray = (data - np.flip(data, axis=lat_axis)) / 2

    return symm_data, asym_data

# chunking data
def chunking(
        data        : np.ndarray,
        time_axis   : int,
        window_size : int,
        overlap_size: int,
) -> np.ndarray:

    """Split data into overlapping, Hann-windowed time segments.

    The returned array has a new leading segment axis. The remaining axes
    retain their input order, with the time axis shortened to ``window_size``.
    Any trailing samples that do not fill a complete segment are discarded.
    """

    # prescribe hanning taper
    hanning_taper: np.ndarray = np.hanning(window_size)

    # calculate step forward
    step: int = window_size - overlap_size

    # starting index for each window
    starts = range(0, data.shape[time_axis] - window_size + 1, step)

    # chunking data
    taper_shape = [1] * data.ndim
    taper_shape[time_axis] = window_size
    hanning_taper = hanning_taper.reshape(taper_shape)

    tmp_chunk = []

    for start in starts:

        indices = [slice(None)] * data.ndim
        indices[time_axis] = slice(start, start + window_size)
        single_chunk: np.ndarray = detrend(data[tuple(indices)], axis=time_axis, type="linear") * hanning_taper

        tmp_chunk.append(single_chunk)

    return np.stack(tmp_chunk)

# ====================================================
# Spectral Analysis
# ====================================================
# Spectrum transform
def space_time_transform(
        data     : np.ndarray,
        time_axis: int,
        lon_axis : int
) -> np.ndarray:
    """Transform data into frequency-zonal-wavenumber space.

    The inverse time FFT and forward longitude FFT preserve the notebook's
    propagation convention. The result is normalized by both dimensions.
    """
    transformed = np.fft.ifft(data, axis=time_axis)
    transformed = np.fft.fft(transformed, axis=lon_axis)
    return transformed / data.shape[lon_axis]

# calculate cross spectrum
def calc_cross_spectrum(
    data1    : np.ndarray,
    data2    : np.ndarray,
    time_axis: int = 0,
    lon_axis : int = 1,
) -> np.ndarray:
    """Return S12 = F(data1) * conj(F(data2))."""
    if data1.shape != data2.shape:
        raise ValueError("Input arrays must have identical shapes.")

    spectrum1 = space_time_transform(data1, time_axis, lon_axis=lon_axis)
    spectrum2 = space_time_transform(data2, time_axis, lon_axis=lon_axis)
    return spectrum1 * np.conj(spectrum2)

def calculate_segment_spectra(
    source_chunks: dict[str, np.ndarray],
    reference_chunks: np.ndarray,
    window_energy: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Calculate latitude-mean cross- and auto-spectra per segment.

    All chunk arrays must be shaped (segment, time, latitude, longitude).
    Processing one segment at a time avoids large four-dimensional FFT
    temporaries. The returned spectral arrays are shaped
    (segment, frequency, zonal_wavenumber).
    """
    if reference_chunks.ndim != 4:
        raise ValueError(
            "reference_chunks must have dimensions "
            "(segment, time, latitude, longitude)."
        )
    if not source_chunks:
        raise ValueError("At least one source field is required.")
    if any(data.shape != reference_chunks.shape for data in source_chunks.values()):
        raise ValueError("All source and reference chunks must have identical shapes.")

    n_segments, n_times, _, n_longitudes = reference_chunks.shape

    if window_energy <= 0:
        raise ValueError("The window must have positive mean-square energy.")

    output_shape = (n_segments, n_times, n_longitudes)
    cross_by_source = {
        name: np.empty(output_shape, dtype=np.complex128)
        for name in source_chunks
    }
    power_by_source = {
        name: np.empty(output_shape, dtype=np.float64)
        for name in source_chunks
    }
    reference_power = np.empty(output_shape, dtype=np.float64)

    for segment in range(n_segments):
        # A segment is shaped (time, latitude, longitude), so the correct
        # transform axes are time=0 and longitude=2.
        reference_fft = space_time_transform(
            reference_chunks[segment], time_axis=0, lon_axis=2
        )
        reference_power[segment] = (
            np.mean(np.abs(reference_fft) ** 2, axis=1) / window_energy
        )

        for name, source in source_chunks.items():
            source_fft = space_time_transform(
                source[segment], time_axis=0, lon_axis=2
            )
            cross_by_source[name][segment] = (
                np.nanmean(source_fft * np.conj(reference_fft), axis=1)
                / window_energy
            )
            power_by_source[name][segment] = (
                np.nanmean(np.abs(source_fft) ** 2, axis=1) / window_energy
            )

    return cross_by_source, power_by_source, reference_power


def calculate_squared_coherence(
    mean_cross_spectrum: np.ndarray,
    mean_source_power: np.ndarray,
    mean_reference_power: np.ndarray,
) -> np.ndarray:
    """Return |<Sxy>|^2 / (<Sxx><Syy>) with safe division."""
    denominator = mean_source_power * mean_reference_power
    coherence = np.divide(
        np.abs(mean_cross_spectrum) ** 2,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return np.clip(coherence, 0.0, 1.0)