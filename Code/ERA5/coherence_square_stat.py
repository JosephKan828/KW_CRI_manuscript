"""Magnitude-squared coherence (MSC) analysis of daily ERA5/CERES fields.

MSC = |mean(X * conj(Y))|**2 / (mean(|X|**2) * mean(|Y|**2)),
where spectra are averaged over latitude and overlapping time segments before
forming the ratio. Each segment is linearly detrended and Hann tapered.
The inverse time FFT and forward longitude FFT preserve the existing wave
propagation convention. Undefined (zero-power) bins remain NaN.

Requires numpy, scipy, scikit-learn, netCDF4, matplotlib, and threadpoolctl.
Configure paths, dates, references, and heating components in the main() call
at the bottom. Shared preprocessing and spectral helpers come from KW_CRI/src.
The MSC estimate explicitly uses spectrum.calculate_squared_coherence().
main() returns full FFT-ordered MSC and segment spectra for later statistics;
only plotting selects positive frequencies. No significance test is applied.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from collections.abc import Sequence
from dataclasses import dataclass

CPU_THREADS = 4
for thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = str(CPU_THREADS)

import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits

THREADPOOL_LIMITER = threadpool_limits(limits=CPU_THREADS)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import preprocessing as preproc
import cross_spectrum_ana as workflow
import spectrum

MeridionalComponent = workflow.MeridionalComponent


def load_field(
    FILE_PATH: Path,
    VARIABLE_KEY: str,
    LATITUDE_BOUNDS: tuple[float, float],
    START_DATE: str | None,
    END_DATE: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a daily field, with latitude sorted south to north.

    START_DATE is inclusive and END_DATE is exclusive, as in Qr_Prec.ipynb.
    W retains its pressure-level axis for the subsequent EOF calculation.
    """
    with nc.Dataset(FILE_PATH) as dataset:
        dates: np.ndarray = preproc.convert_time(dataset.variables["time"])
        latitude: np.ndarray = preproc.as_float_array(dataset.variables["lat"][:])
        longitude: np.ndarray = preproc.as_float_array(dataset.variables["lon"][:])
        date_mask: np.ndarray = np.ones(dates.size, dtype=bool)
        
        if START_DATE is not None:
            date_mask &= dates >= START_DATE
        if END_DATE is not None:
            date_mask &= dates < END_DATE
        
        latitude_mask: np.ndarray = (latitude >= LATITUDE_BOUNDS[0]) & (latitude <= LATITUDE_BOUNDS[1])

        variable = dataset.variables[VARIABLE_KEY]
        if VARIABLE_KEY == "W":
            field: np.ndarray = preproc.as_float_array(variable[date_mask, :, latitude_mask, :])
        else:
            field = preproc.as_float_array(variable[date_mask, latitude_mask, :])

    latitude_order: np.ndarray = np.argsort(latitude[latitude_mask])
    return (
        np.take(field, latitude_order, axis=-2),
        dates[date_mask],
        latitude[latitude_mask][latitude_order],
        longitude,
    )


def load_analysis_fields(
    DATA_PATH: Path,
    LATITUDE_BOUNDS: tuple[float, float],
    REFERENCE_KEY: str,
    SOURCE_NAME: str,
    START_DATE: str | None,
    END_DATE: str | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load selected heating and reference fields and match their calendar days."""
    lw_heating, dates, latitude, longitude = load_field(
        DATA_PATH / "CERES" / "column_LW_rm3harm.nc", "column_LW",
        LATITUDE_BOUNDS, START_DATE, END_DATE,
    )
    sw_heating, sw_dates, sw_latitude, sw_longitude = load_field(
        DATA_PATH / "CERES" / "column_SW_rm3harm.nc", "column_SW",
        LATITUDE_BOUNDS, START_DATE, END_DATE,
    )

    reference_file: Path = DATA_PATH / "ERA5_GRIB" / "Data" / "regrid" / f"{REFERENCE_KEY}.nc"
    
    reference_field, reference_dates, reference_latitude, reference_longitude = load_field(
        reference_file, REFERENCE_KEY, LATITUDE_BOUNDS, START_DATE, END_DATE,
    )

    for name, lat, lon in (
        ("SW", sw_latitude, sw_longitude),
        (REFERENCE_KEY, reference_latitude, reference_longitude),
    ):
        if (lat.shape != latitude.shape or lon.shape != longitude.shape
                or not np.allclose(lat, latitude) or not np.allclose(lon, longitude)):
            raise ValueError(f"{name} and LW must use the same latitude/longitude grid.")

    # Match all three calendars before removing means or calculating omega EOFs.
    common_dates = np.intersect1d(np.intersect1d(dates, sw_dates), reference_dates)
    if common_dates.size == 0:
        raise ValueError("No common dates exist in the selected interval.")
    if np.any(np.diff(common_dates.astype("datetime64[D]")) != np.timedelta64(1, "D")):
        raise ValueError("The common dates must form a continuous daily record.")

    def align(field: np.ndarray, field_dates: np.ndarray) -> np.ndarray:
        if np.unique(field_dates).size != field_dates.size:
            raise ValueError("Input fields must contain only one sample per calendar day.")
        _, indices, _ = np.intersect1d(field_dates, common_dates, return_indices=True)
        return field[indices]

    lw_heating = align(lw_heating, dates)
    sw_heating = align(sw_heating, sw_dates)
    reference_field = align(reference_field, reference_dates)
    print(f"Analyzing {common_dates[0]} through {common_dates[-1]} ({common_dates.size} days).")

    source_fields: dict[str, np.ndarray] = {}
    if SOURCE_NAME in {"QR", "ALL"}:
        source_fields["QR"] = lw_heating + sw_heating
    if SOURCE_NAME in {"LW", "ALL"}:
        source_fields["LW"] = lw_heating
    if SOURCE_NAME in {"SW", "ALL"}:
        source_fields["SW"] = sw_heating
    return source_fields, reference_field


def prepare_components(
    source_fields: dict[str, np.ndarray],
    reference_field: np.ndarray,
    REFERENCE_KEY: str,
    MERIDIONAL_COMPONENT: MeridionalComponent,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Remove means, derive omega PCs when selected, and select meridional components."""
    source_anomalies: dict[str, np.ndarray] = {name: workflow.remove_time_zonal_mean(field) for name, field in source_fields.items()}

    reference_anomalies: dict[str, np.ndarray]

    if REFERENCE_KEY in {"W", "W1", "W2"}:
        
        nt, nz, ny, nx = reference_field.shape # assigning fields

        # calculate zonal and temporal anomalies
        omega_anom: np.ndarray = reference_field - np.nanmean(reference_field, axis=(0, -1), keepdims=True)

        # calculate EOFs of the omega anomalies
        _, w_pcs = preproc.calculate_eof(
            omega_anom.transpose(1, 0, 2, 3).reshape(nz, -1),
            n_components=2,
        )

        reference_anomalies = {
            f"W{i+1}": -w_pcs[i].reshape(nt, ny, nx)
            for i in range(w_pcs.shape[0])
        }
        if REFERENCE_KEY != "W":
            reference_anomalies = {REFERENCE_KEY: reference_anomalies[REFERENCE_KEY]}
        del omega_anom

    else:
        reference_anomalies = {REFERENCE_KEY: workflow.remove_time_zonal_mean(reference_field)}

    source_components: dict[str, np.ndarray] = {
        name: workflow.select_meridional_component(field, MERIDIONAL_COMPONENT)
        for name, field in source_anomalies.items()
    }
    reference_components: dict[str, np.ndarray] = {
        name: workflow.select_meridional_component(field, MERIDIONAL_COMPONENT)
        for name, field in reference_anomalies.items()
    }
    for reference_component in reference_components.values():
        workflow.validate_aligned_fields(source_components, reference_component)
    return source_components, reference_components


@dataclass
class CoherenceResult:
    """MSC and its segment spectra for one source/reference pair.

    Coordinates use full, unshifted FFT order. MSC has shape (frequency,
    wavenumber); spectra have shape (segment, frequency, wavenumber), with
    latitude already averaged. Segment starts index the aligned daily data.
    Overlapping segments are dependent; their count is not an effective
    number of independent samples for a later significance test.
    """

    frequencies: np.ndarray
    zonal_wavenumbers: np.ndarray
    coherence_squared: np.ndarray
    cross_spectrum_segments: np.ndarray
    source_power_segments: np.ndarray
    reference_power_segments: np.ndarray
    segment_start_indices: np.ndarray
    window_size: int
    overlap_size: int


def calculate_coherence(
    source_fields: dict[str, np.ndarray],
    reference_field: np.ndarray,
    window_size: int,
    overlap_size: int,
) -> dict[str, CoherenceResult]:
    """Use shared helpers and explicitly evaluate the MSC estimator.

    Keep the segment spectra so later resampling can average complex cross-
    spectra and auto-spectra, then call calculate_squared_coherence() again.
    Averaging coherence values from individual segments is a different estimator.
    """
    workflow.validate_aligned_fields(source_fields, reference_field)
    if window_size < 3:
        raise ValueError("window_size must be at least 3 for a nonzero Hann taper.")
    source_chunks = workflow.segment_fields(source_fields, window_size, overlap_size)
    reference_chunks = workflow.segment_fields(
        {"reference": reference_field}, window_size, overlap_size,
    )["reference"]
    cross_spectra, source_power, reference_power = workflow.calculate_segment_spectra(
        source_chunks, reference_chunks, window_size=window_size,
    )
    del source_chunks, reference_chunks
    mean_cross, mean_source_power, mean_reference_power = workflow.average_segment_spectra(
        cross_spectra, source_power, reference_power,
    )
    frequencies, wavenumbers = workflow.spectral_coordinates(
        window_size, reference_field.shape[-1], sampling_interval=1.0,
    )
    starts = np.arange(
        0, reference_field.shape[0] - window_size + 1, window_size - overlap_size,
    )
    results = {}
    for source_name in source_fields:
        # Explicit MSC call: |<Sxy>|² / (<Sxx> <Syy>).
        coherence_squared = spectrum.calculate_squared_coherence(
            mean_cross_spectrum=mean_cross[source_name],
            mean_source_power=mean_source_power[source_name],
            mean_reference_power=mean_reference_power,
        )
        results[source_name] = CoherenceResult(
            frequencies=frequencies,
            zonal_wavenumbers=wavenumbers,
            coherence_squared=coherence_squared,
            cross_spectrum_segments=cross_spectra[source_name],
            source_power_segments=source_power[source_name],
            reference_power_segments=reference_power,
            segment_start_indices=starts,
            window_size=window_size,
            overlap_size=overlap_size,
        )
    print(f"Averaged {starts.size} segments of {window_size} days.")
    return results


def analyze_spectra(
    source_components: dict[str, np.ndarray],
    reference_components: dict[str, np.ndarray],
    WINDOW_SIZE: int,
    OVERLAP_SIZE: int,
    EQUIVALENT_DEPTHS: tuple[float, float],
) -> dict[str, dict[str, CoherenceResult]]:
    """Return MSC and segment spectra indexed by reference, then source name."""
    results = {}
    for reference_name, reference in reference_components.items():
        results[reference_name] = calculate_coherence(
            source_components, reference, WINDOW_SIZE, OVERLAP_SIZE,
        )
        for source_name, result in results[reference_name].items():
            statistics = workflow.average_wave_domain_coherence(
                result.zonal_wavenumbers, result.frequencies, result.coherence_squared,
                equivalent_depths=EQUIVALENT_DEPTHS,
            )
            print(f"\n{source_name} / {reference_name}")
            print(f"{'Domain':<10} {'Mean coherence²':>17} {'Finite bins':>12}")
            for domain, stats in statistics.items():
                print(f"{domain:<10} {stats['mean_coherence_squared']:>17.4f} {stats['n_bins']:>12d}")
    return results


def plot_coherence(
    zonal_wavenumbers: np.ndarray,
    frequencies: np.ndarray,
    coherence_squared: np.ndarray,
    *,
    shading_levels: Sequence[float] = tuple(np.linspace(0.0, 1.0, 21)),
    coherence_levels: Sequence[float] = (0.3, 0.5, 0.7),
    equivalent_depths: Sequence[float] = (8.0, 90.0),
    show_wave_domains: bool = True,
    cmap: str = "viridis",
    colorbar_label: str = "Magnitude-squared coherence ($\\gamma^2$)",
    x_limits: tuple[float, float] = (-15.0, 15.0),
    y_limits: tuple[float, float] = (0.0, 0.5),
    figure_size: tuple[float, float] = (8.0, 6.0),
    axis: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot MSC shading on [0, 1], optional contours, and wave-domain means."""
    if coherence_squared.shape != (frequencies.size, zonal_wavenumbers.size):
        raise ValueError("coherence_squared must have shape (frequency, wavenumber).")

    domain_means = workflow.average_wave_domain_coherence(
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
        coherence_squared,
        levels=shading_levels,
        cmap=cmap,
        extend="neither",
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
            colors="white",
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
        for name, k_bounds, f_bounds, branch in workflow._WAVE_DOMAINS:
            # A dense grid follows the curved boundary independently of FFT bins.
            domain_k = np.linspace(*k_bounds, 1001)
            if branch is None:
                lower = np.full_like(domain_k, f_bounds[0])
                upper = np.full_like(domain_k, f_bounds[1])
            else:
                curves = [workflow.dispersion_relation(domain_k, depth)[branch] for depth in depths]
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
            "Mean magnitude-squared coherence\n\n"
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


def plot_spectra(
    results: dict[str, dict[str, CoherenceResult]],
    OUTPUT_PATH: Path,
    REFERENCE_KEY: str,
    MERIDIONAL_COMPONENT: MeridionalComponent,
    COHERENCE_LEVELS: tuple[float, ...],
    EQUIVALENT_DEPTHS: tuple[float, float],
    FIGURE_DPI: int,
    SHOW_FIGURES: bool,
) -> None:
    """Save one figure per heating component, with paired panels for W1 and W2."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 14, "axes.labelsize": 14,
        "axes.titlesize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    })
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    reference_names: tuple[str, ...] = tuple(results)
    source_names: tuple[str, ...] = tuple(results[reference_names[0]])
    for source_name in source_names:
        n_panels: int = len(reference_names)
        figure, axes = plt.subplots(
            1, n_panels, figsize=(12, 5) if n_panels == 2 else (8, 6),
            sharex=True, sharey=True, constrained_layout=True, squeeze=False,
        )
        for axis, reference_name in zip(axes[0], reference_names):
            result = results[reference_name][source_name]
            frequencies = np.fft.fftshift(result.frequencies)
            wavenumbers = np.fft.fftshift(result.zonal_wavenumbers)
            positive = frequencies > 0
            coherence_plot = np.fft.fftshift(result.coherence_squared, axes=(0, 1))[positive]
            plot_coherence(
                wavenumbers, frequencies[positive], coherence_plot,
                shading_levels=np.linspace(0.0, 1.0, 21), #type: ignore
                coherence_levels=COHERENCE_LEVELS,
                equivalent_depths=EQUIVALENT_DEPTHS,
                axis=axis,
            )
            axis.set_title(f"{source_name} / {reference_name} ({MERIDIONAL_COMPONENT})")
        figure_path: Path = OUTPUT_PATH / f"{source_name}_{REFERENCE_KEY}_{MERIDIONAL_COMPONENT}_MSC.png"
        figure.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved {figure_path}")
        if SHOW_FIGURES:
            plt.show()
        plt.close(figure)


def main(
    DATA_PATH: Path,
    OUTPUT_PATH: Path,
    LATITUDE_BOUNDS: tuple[float, float],
    WINDOW_SIZE: int,
    OVERLAP_SIZE: int,
    MERIDIONAL_COMPONENT: MeridionalComponent,
    REFERENCE_KEY: str,
    SOURCE_NAME: str = "QR",
    START_DATE: str | None = "2006-01-01",
    END_DATE: str | None = "2018-01-01",
    COHERENCE_LEVELS: tuple[float, ...] | None = None,
    EQUIVALENT_DEPTHS: tuple[float, float] = (8.0, 90.0),
    FIGURE_DPI: int = 600,
    SHOW_FIGURES: bool = False,
) -> dict[str, dict[str, CoherenceResult]]:
    """Run loading, preprocessing, spectral analysis, and plotting in sequence.

    REFERENCE_KEY: "CWV", "precipitation" (or "Prec"), "W", "W1", or "W2".
    "W" (or "omega") plots both negative omega PCs side by side.
    SOURCE_NAME: "QR" (LW + SW), "LW", "SW", or "ALL" for all three figures.
    EOF normalization and PC signs follow Qr_omega.ipynb unchanged.
    Inputs are assumed to have complete daily records on the prepared grids.
    Returns results[reference_name][source_name], e.g. results["CWV"]["QR"].
    Each CoherenceResult exposes coherence_squared, coordinates, segment spectra,
    segment_start_indices, window_size, and overlap_size for further statistics.
    """
    reference_aliases: dict[str, str] = {
        "cwv": "CWV", "prec": "precipitation", "precipitation": "precipitation",
        "omega": "W", "w": "W", "w1": "W1", "w2": "W2",
    }
    if REFERENCE_KEY.lower() not in reference_aliases:
        raise ValueError(f"Unsupported reference: {REFERENCE_KEY!r}.")
    REFERENCE_KEY = reference_aliases[REFERENCE_KEY.lower()]
    SOURCE_NAME = SOURCE_NAME.upper()
    if SOURCE_NAME not in {"QR", "LW", "SW", "ALL"}:
        raise ValueError("SOURCE_NAME must be QR, LW, SW, or ALL.")
    if COHERENCE_LEVELS is None:
        COHERENCE_LEVELS = (0.3, 0.4, 0.5, 0.6, 0.8) if REFERENCE_KEY == "CWV" else (0.5, 0.6, 0.8)

    # Load heating and reference fields on common dates.
    source_fields, reference_field = load_analysis_fields(
        DATA_PATH, LATITUDE_BOUNDS, "W" if REFERENCE_KEY in {"W1", "W2"} else REFERENCE_KEY,
        SOURCE_NAME, START_DATE, END_DATE,
    )

    # Remove means and select symmetric, antisymmetric, or full fields.
    source_components, reference_components = prepare_components(
        source_fields, reference_field, REFERENCE_KEY, MERIDIONAL_COMPONENT,
    )
    del source_fields, reference_field

    # Calculate magnitude-squared coherence.
    results = analyze_spectra(
        source_components, reference_components, WINDOW_SIZE, OVERLAP_SIZE, EQUIVALENT_DEPTHS,
    )

    # Save figures for the selected heating components.
    plot_spectra(
        results, OUTPUT_PATH, REFERENCE_KEY, MERIDIONAL_COMPONENT,
        COHERENCE_LEVELS, EQUIVALENT_DEPTHS, FIGURE_DPI, SHOW_FIGURES,
    )
    return results


if __name__ == "__main__":
    # ALL produces separate QR, LW, and SW figures while loading each reference once.
    results = {}
    for reference_key in ("CWV", "W"):
        results.update(main(
            DATA_PATH=Path("/data92/b11209013"),
            OUTPUT_PATH=Path(__file__).resolve().parents[2] / "Figure" / "New_analysis_Figure" / "Magnitude_Squared_Coherence",
            LATITUDE_BOUNDS=(-5.0, 5.0),
            WINDOW_SIZE=96,
            OVERLAP_SIZE=48,
            MERIDIONAL_COMPONENT="symmetric",
            REFERENCE_KEY=reference_key,  # CWV, precipitation (or Prec), W, W1, W2
            SOURCE_NAME="ALL",  # QR = LW + SW; LW, SW, or ALL
            START_DATE="2006-01-01",  # Inclusive; None uses all available dates
            END_DATE="2018-01-01",  # Exclusive; None uses all available dates
            COHERENCE_LEVELS=None,  # Optional contour levels; () hides contours
            EQUIVALENT_DEPTHS=(8.0, 90.0),
            FIGURE_DPI=300,
            SHOW_FIGURES=False,
        ))

    # Available for subsequent statistical analysis in an interactive session:
    # results["CWV"]["QR"].coherence_squared
    # results["CWV"]["QR"].cross_spectrum_segments
    # results["W1"]["LW"].source_power_segments
