"""Function-based workflow for repeated cross-spectral analyses.

Load and align data in a notebook, then apply these functions in sequence:

1. :func:`remove_time_zonal_mean`
2. :func:`select_meridional_component`
3. :func:`validate_aligned_fields`
4. :func:`segment_fields`
5. :func:`calculate_segment_spectra`
6. :func:`average_segment_spectra`
7. :func:`calculate_spectral_diagnostics`
8. :func:`spectral_coordinates`
9. :func:`prepare_plot_data`
10. :func:`plot_cospectrum_with_coherence`

The steps remain separate so a notebook clearly shows where preprocessing,
spectral estimation, normalization, and plotting choices can be changed.
All input fields are expected to be ordered as (time, latitude, longitude).
"""

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import spectrum


MeridionalComponent = Literal["symmetric", "antisymmetric", "none"]

__all__ = [
    "remove_time_zonal_mean",
    "select_meridional_component",
    "validate_aligned_fields",
    "segment_fields",
    "calculate_segment_spectra",
    "average_segment_spectra",
    "calculate_spectral_diagnostics",
    "spectral_coordinates",
    "prepare_plot_data",
    "average_wave_domain_coherence",
    "plot_cospectrum_with_coherence",
]


# Build constant
gravity = 9.81
seconds_per_day = 86400.0
earth_radius = 6.371e6
Omega = 7.292e-5

# name, zonal-wavenumber bounds, frequency bounds (cycles/day), branch index
# A None branch denotes a rectangle without dispersion-curve clipping.
_WAVE_DOMAINS = (
    ("Kelvin", (3.0, 8.0), (1.0 / 20.0, 0.3), 0),
    ("Rossby", (-8.0, -2.0), (0.02, 0.1), 1),
    ("MJO", (1.0, 3.0), (1.0 / 90.0, 1.0 / 30.0), None),
)

def remove_time_zonal_mean(
    data: np.ndarray,
    time_axis: int = 0,
    longitude_axis: int = -1,
) -> np.ndarray:
    """Remove the time-zonal mean while retaining the latitude structure."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            "data must have dimensions (time, latitude, longitude)."
        )
    if time_axis % values.ndim == longitude_axis % values.ndim:
        raise ValueError("time_axis and longitude_axis must be distinct.")
    return values - np.nanmean(
        values,
        axis=(time_axis, longitude_axis),
        keepdims=True,
    )


def select_meridional_component(
    data: np.ndarray,
    component: MeridionalComponent = "symmetric",
    latitude_axis: int = 1,
) -> np.ndarray:
    """Return the symmetric, antisymmetric, or unchanged meridional field.

    Symmetric and antisymmetric decomposition assumes that latitude indices
    are paired about the equator.
    """
    values = np.asarray(data)
    if values.ndim != 3:
        raise ValueError(
            "data must have dimensions (time, latitude, longitude)."
        )
    if component == "none":
        return np.array(values, copy=True)
    if component not in {"symmetric", "antisymmetric"}:
        raise ValueError(
            "component must be 'symmetric', 'antisymmetric', or 'none'."
        )

    symmetric, antisymmetric = spectrum.symm_asym(
        values,
        lat_axis=latitude_axis,
    )
    return symmetric if component == "symmetric" else antisymmetric


def validate_aligned_fields(
    source_fields: Mapping[str, np.ndarray],
    reference_field: np.ndarray,
) -> None:
    """Validate array properties needed by the spectral calculation.

    This checks shapes and finite values. Dates, latitudes, and longitudes must
    still be compared in the notebook because arrays do not carry coordinate
    metadata.
    """
    if not source_fields:
        raise ValueError("At least one source field is required.")

    reference = np.asarray(reference_field)
    if reference.ndim != 3:
        raise ValueError(
            "reference_field must have dimensions (time, latitude, longitude)."
        )
    if not np.all(np.isfinite(reference)):
        raise ValueError(
            "reference_field contains NaN or infinite values; detrending "
            "requires finite input."
        )

    for name, field in source_fields.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Every source field must have a nonempty name.")
        values = np.asarray(field)
        if values.shape != reference.shape:
            raise ValueError(
                f"Source {name!r} has shape {values.shape}; expected "
                f"{reference.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Source {name!r} contains NaN or infinite values; "
                "detrending requires finite input."
            )


def segment_fields(
    fields: Mapping[str, np.ndarray],
    window_size: int,
    overlap_size: int,
) -> dict[str, np.ndarray]:
    """Detrend and Hann-window all fields into overlapping time segments."""
    if not fields:
        raise ValueError("At least one field is required.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if not 0 <= overlap_size < window_size:
        raise ValueError(
            "overlap_size must satisfy 0 <= overlap_size < window_size."
        )

    chunks: dict[str, np.ndarray] = {}
    for name, field in fields.items():
        values = np.asarray(field)
        if values.ndim != 3:
            raise ValueError(
                f"Field {name!r} must have dimensions "
                "(time, latitude, longitude)."
            )
        if values.shape[0] < window_size:
            raise ValueError(
                f"Field {name!r} has only {values.shape[0]} time samples, "
                f"fewer than window_size={window_size}."
            )
        chunks[name] = spectrum.chunking(
            values,
            time_axis=0,
            window_size=window_size,
            overlap_size=overlap_size,
        )
    return chunks


def calculate_segment_spectra(
    source_chunks: Mapping[str, np.ndarray],
    reference_chunks: np.ndarray,
    window_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Calculate cross- and auto-spectra for every windowed segment."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    window_energy = float(np.mean(np.hanning(window_size) ** 2))
    return spectrum.calculate_segment_spectra(
        source_chunks=dict(source_chunks),
        reference_chunks=reference_chunks,
        window_energy=window_energy,
    )


def average_segment_spectra(
    cross_spectrum: Mapping[str, np.ndarray],
    source_power_spectrum: Mapping[str, np.ndarray],
    reference_power_spectrum: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Average segment-level spectra, preserving frequency and wavenumber."""
    if cross_spectrum.keys() != source_power_spectrum.keys():
        raise ValueError(
            "Cross-spectrum and source-power dictionaries must have the same keys."
        )

    mean_cross_spectrum = {
        name: np.nanmean(values, axis=0)
        for name, values in cross_spectrum.items()
    }
    mean_source_power_spectrum = {
        name: np.nanmean(values, axis=0)
        for name, values in source_power_spectrum.items()
    }
    mean_reference_power_spectrum = np.nanmean(
        reference_power_spectrum,
        axis=0,
    )
    return (
        mean_cross_spectrum,
        mean_source_power_spectrum,
        mean_reference_power_spectrum,
    )


def calculate_spectral_diagnostics(
    mean_cross_spectrum: Mapping[str, np.ndarray],
    mean_source_power_spectrum: Mapping[str, np.ndarray],
    mean_reference_power_spectrum: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Derive co-spectrum, coherence, phase, and related diagnostics."""
    if mean_cross_spectrum.keys() != mean_source_power_spectrum.keys():
        raise ValueError(
            "Mean cross-spectrum and source-power dictionaries must share keys."
        )

    cospectrum = {
        name: np.real(values)
        for name, values in mean_cross_spectrum.items()
    }
    quadrature_spectrum = {
        name: np.imag(values)
        for name, values in mean_cross_spectrum.items()
    }
    normalized_cospectrum = {
        name: spectrum.calculate_normalized_cospectrum(
            mean_cross_spectrum=mean_cross_spectrum[name],
            mean_source_power=mean_source_power_spectrum[name],
            mean_reference_power=mean_reference_power_spectrum,
        )
        for name in mean_cross_spectrum
    }
    coherence_squared = {
        name: spectrum.calculate_squared_coherence(
            mean_cross_spectrum=mean_cross_spectrum[name],
            mean_source_power=mean_source_power_spectrum[name],
            mean_reference_power=mean_reference_power_spectrum,
        )
        for name in mean_cross_spectrum
    }
    phase_degrees = {
        name: np.rad2deg(np.angle(values))
        for name, values in mean_cross_spectrum.items()
    }

    return {
        "cospectrum": cospectrum,
        "quadrature_spectrum": quadrature_spectrum,
        "normalized_cospectrum": normalized_cospectrum,
        "coherence_squared": coherence_squared,
        "phase_degrees": phase_degrees,
    }


def spectral_coordinates(
    window_size: int,
    n_longitudes: int,
    sampling_interval: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return FFT frequency and integer zonal-wavenumber coordinates."""
    if window_size <= 0 or n_longitudes <= 0:
        raise ValueError("window_size and n_longitudes must be positive.")
    if sampling_interval <= 0:
        raise ValueError("sampling_interval must be positive.")

    frequencies = np.fft.fftfreq(window_size, d=sampling_interval)
    zonal_wavenumbers = np.fft.fftfreq(
        n_longitudes,
        d=1.0 / n_longitudes,
    )
    return frequencies, zonal_wavenumbers


def prepare_plot_data(
    frequencies: np.ndarray,
    zonal_wavenumbers: np.ndarray,
    cospectrum: np.ndarray,
    coherence_squared: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """FFT-shift spectra and retain positive temporal frequencies."""
    expected_shape = (frequencies.size, zonal_wavenumbers.size)
    if cospectrum.shape != expected_shape:
        raise ValueError(
            f"cospectrum has shape {cospectrum.shape}; expected {expected_shape}."
        )
    if coherence_squared.shape != expected_shape:
        raise ValueError(
            "coherence_squared must have the same shape as cospectrum."
        )

    shifted_frequencies = np.fft.fftshift(frequencies)
    shifted_wavenumbers = np.fft.fftshift(zonal_wavenumbers)
    shifted_cospectrum = np.fft.fftshift(cospectrum, axes=(0, 1))
    shifted_coherence = np.fft.fftshift(coherence_squared, axes=(0, 1))

    positive_frequency = shifted_frequencies > 0
    return (
        shifted_wavenumbers,
        shifted_frequencies[positive_frequency],
        shifted_cospectrum[positive_frequency, :],
        shifted_coherence[positive_frequency, :],
    )


def dispersion_relation(
        zonal_wavenumbers: np.ndarray,
        eqilivalent_depth: float
) -> tuple[np.ndarray, ...]:
    """Return Kelvin and n=1 equatorial Rossby frequencies in cycles/day.

    Zonal wavenumbers count wavelengths around the equator; equivalent depth
    is in metres. The Rossby relation uses the low-frequency approximation
    with no background zonal flow.
    """

    # Kelvin waves
    KW_dispersion: np.ndarray = (
            np.sqrt(gravity * eqilivalent_depth)
            * seconds_per_day
            / (2.0 * np.pi * earth_radius)
            * zonal_wavenumbers
        )

    # Equatorial Rossby waves, meridional mode n=1.
    c = np.sqrt(gravity * eqilivalent_depth)
    beta = 2.0 * Omega / earth_radius
    k = np.asarray(zonal_wavenumbers, dtype=float) / earth_radius
    n = 1
    RW_dispersion: np.ndarray = (
        -beta * k / (k**2 + (2 * n + 1) * beta / c)
        * seconds_per_day / (2.0 * np.pi)
    )

    return KW_dispersion, RW_dispersion


def average_wave_domain_coherence(
    zonal_wavenumbers: np.ndarray,
    frequencies: np.ndarray,
    coherence_squared: np.ndarray,
    *,
    equivalent_depths: Sequence[float] = (8.0, 90.0),
) -> dict[str, dict[str, float | int]]:
    """Average squared coherence over the domains defined in _WAVE_DOMAINS.

    Kelvin and Rossby (n=1) rectangles are intersected with the band between
    the two equivalent-depth dispersion curves (depths in metres). The
    MJO domain is a rectangle with k in [1, 3] and frequency in
    [1/90, 1/30] cycles/day, without dispersion clipping. Bounds are inclusive.

    Coordinates may be FFT-ordered or shifted; coherence must have shape
    (frequency, wavenumber) in the same order. This is an unweighted mean of
    finite squared-coherence bins, not coherence computed from pooled spectra.
    Empty domains return NaN and n_bins=0; n_bins counts finite values only.
    """
    wavenumbers = np.asarray(zonal_wavenumbers, dtype=float)
    frequency = np.asarray(frequencies, dtype=float)
    values = np.asarray(coherence_squared, dtype=float)
    depths = np.asarray(equivalent_depths, dtype=float)
    if wavenumbers.ndim != 1 or frequency.ndim != 1:
        raise ValueError("Wavenumber and frequency coordinates must be 1D.")
    if not np.all(np.isfinite(wavenumbers)) or not np.all(np.isfinite(frequency)):
        raise ValueError("Wavenumber and frequency coordinates must be finite.")
    if values.shape != (frequency.size, wavenumbers.size):
        raise ValueError("coherence_squared must have shape (frequency, wavenumber).")
    if depths.shape != (2,) or not np.all(np.isfinite(depths)) or np.any(depths <= 0):
        raise ValueError("equivalent_depths must contain two finite positive depths.")

    curves = [dispersion_relation(wavenumbers, depth) for depth in depths]
    results: dict[str, dict[str, float | int]] = {}
    for name, k_bounds, f_bounds, branch in _WAVE_DOMAINS:
        mask = (
            (wavenumbers[None, :] >= k_bounds[0])
            & (wavenumbers[None, :] <= k_bounds[1])
            & (frequency[:, None] >= f_bounds[0])
            & (frequency[:, None] <= f_bounds[1])
            & np.isfinite(values)
        )
        if branch is not None:
            lower = np.minimum(curves[0][branch], curves[1][branch])
            upper = np.maximum(curves[0][branch], curves[1][branch])
            mask &= (
                (frequency[:, None] >= lower[None, :])
                & (frequency[:, None] <= upper[None, :])
            )
        selected = values[mask]
        results[name] = {
            "mean_coherence_squared": float(selected.mean()) if selected.size else float("nan"),
            "n_bins": int(selected.size),
        }
    return results


def plot_cospectrum_with_coherence(
    zonal_wavenumbers: np.ndarray,
    frequencies: np.ndarray,
    cospectrum: np.ndarray,
    coherence_squared: np.ndarray,
    *,
    source_name: str,
    reference_name: str,
    shading_levels: Sequence[float] = tuple(np.linspace(-1.0, 1.0, 21)),
    coherence_levels: Sequence[float] = (0.3, 0.5, 0.7),
    equivalent_depths: Sequence[float] = (8.0, 90.0),
    show_wave_domains: bool = True,
    cmap: str = "RdBu_r",
    extend: str = "both",
    colorbar_label: str = "Normalized co-spectrum",
    x_limits: tuple[float, float] = (-15.0, 15.0),
    y_limits: tuple[float, float] = (0.0, 0.5),
    figure_size: tuple[float, float] = (8.0, 6.0),
    axis: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a co-spectrum with squared-coherence contours.

    The function returns the figure and axes without displaying or saving
    them, leaving those choices explicit in the calling notebook.
    By default, solid frames outline the Kelvin and n=1 Rossby averaging
    domains between two equivalent-depth curves and the specified domain
    limits, plus the rectangular MJO domain without dispersion clipping.
    """
    expected_shape = (frequencies.size, zonal_wavenumbers.size)
    if cospectrum.shape != expected_shape:
        raise ValueError(
            f"cospectrum has shape {cospectrum.shape}; expected {expected_shape}."
        )
    if coherence_squared.shape != expected_shape:
        raise ValueError(
            "coherence_squared must have the same shape as cospectrum."
        )

    domain_means = average_wave_domain_coherence(
        zonal_wavenumbers,
        frequencies,
        coherence_squared,
        equivalent_depths=equivalent_depths,
    )

    if show_wave_domains:
        depths = np.asarray(equivalent_depths, dtype=float)
        if depths.shape != (2,) or not np.all(np.isfinite(depths)) or np.any(depths <= 0):
            raise ValueError("Domain frames require two finite positive equivalent depths.")

    if axis is None:
        figure, axis = plt.subplots(
            figsize=figure_size,
            constrained_layout=True,
        )
    else:
        figure = axis.figure
    shading = axis.contourf(
        zonal_wavenumbers,
        frequencies,
        cospectrum,
        levels=shading_levels,
        cmap=cmap,
        extend=extend,
        alpha=0.5
    )

    finite_coherence = coherence_squared[np.isfinite(coherence_squared)]
    available_levels: list[float] = []
    if finite_coherence.size:
        available_levels = [
            level
            for level in coherence_levels
            if finite_coherence.min() < level < finite_coherence.max()
        ]
    if available_levels:
        contours = axis.contour(
            zonal_wavenumbers,
            frequencies,
            coherence_squared,
            levels=available_levels,
            colors="green",
            linewidths=1.5,
        )
        axis.clabel(
            contours,
            fmt=lambda value: rf"$\gamma^2={value:.1f}$",
            fontsize=12,
            inline=True,
        )

    if show_wave_domains:
        domain_colors = {"Kelvin": "k", "Rossby": "k", "MJO": "k"}
        for name, k_bounds, f_bounds, branch in _WAVE_DOMAINS:
            # A dense grid follows the curved boundary independently of FFT bins.
            domain_k = np.linspace(*k_bounds, 1001)
            if branch is None:
                lower = np.full_like(domain_k, f_bounds[0])
                upper = np.full_like(domain_k, f_bounds[1])
            else:
                curves = [dispersion_relation(domain_k, depth)[branch] for depth in depths]
                lower = np.maximum(np.minimum(curves[0], curves[1]), f_bounds[0])
                upper = np.minimum(np.maximum(curves[0], curves[1]), f_bounds[1])
            if not np.any(lower < upper):
                continue
            axis.fill_between(
                domain_k,
                lower,
                upper,
                where=lower <= upper, #type: ignore
                interpolate=True,
                facecolor="none",
                edgecolor=domain_colors[name],
                linewidth=2.5,
                linestyle="--",
                zorder=5,
            )
        domain_summary = (
            "Mean coherence square\n\n"
            f"$\\mathbf{{Kelvin}}$:{domain_means['Kelvin']['mean_coherence_squared']:.3f}\n"
            f"$\\mathbf{{Rossby}}$:{domain_means['Rossby']['mean_coherence_squared']:.3f}\n"
            f"$\\mathbf{{MJO}}$:{domain_means['MJO']['mean_coherence_squared']:.3f}"
        )
        axis.text(
            0.98,
            0.98,
            domain_summary,
            transform=axis.transAxes,
            ha="right",
            multialignment="left",
            va="top",
            fontsize=10,
            fontfamily="monospace",
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.85},
            zorder=6,
        )

    axis.set(
        xlim=x_limits,
        ylim=y_limits,
        xlabel="Zonal wavenumber",
        ylabel="Frequency (cycles day$^{-1}$)",
        title="",
    )
    figure.colorbar(shading, ax=axis, label=colorbar_label)
    return figure, axis
