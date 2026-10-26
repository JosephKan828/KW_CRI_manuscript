"""
    This script performs spectral analysis on ERA5 and Satellite data,
    including normalized co-spectrum and magnitude-squared coherence (MSC)
    calculations. Configure the reference and heating component in the
    main() call at the bottom of this file.
"""

# ====================================================
# Import package and necessary configurations
# ====================================================

import os
import sys

CPU_THREADS = 4
for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = str(CPU_THREADS)

from pathlib import Path
from datetime import datetime

import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from threadpoolctl import threadpool_limits

# Also enforce the limit if a numerical library was loaded by the kernel.
THREADPOOL_LIMITER = threadpool_limits(limits=CPU_THREADS)

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
import cross_spectrum_ana as workflow #type: ignore
import preprocessing as preproc       #type: ignore

# ====================================================
# Data loading functions
# ====================================================

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
        dates: np.ndarray = convert_time(dataset.variables["time"])
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
    lw_heating, dates, _, _ = load_field(
        DATA_PATH / "CERES" / "column_LW_rm3harm.nc", "column_LW",
        LATITUDE_BOUNDS, START_DATE, END_DATE,
    )
    sw_heating, _, _, _ = load_field(
        DATA_PATH / "CERES" / "column_SW_rm3harm.nc", "column_SW",
        LATITUDE_BOUNDS, START_DATE, END_DATE,
    )

    reference_file: Path = DATA_PATH / "ERA5_GRIB" / "Data" / "regrid" / f"{REFERENCE_KEY}.nc"
    
    reference_field, reference_dates, _, _ = load_field(
        reference_file, REFERENCE_KEY, LATITUDE_BOUNDS, START_DATE, END_DATE,
    )

    # Match calendar days before removing means or calculating omega EOFs.
    common_dates, source_indices, reference_indices = np.intersect1d(
        dates, reference_dates, return_indices=True,
    )
    lw_heating = lw_heating[source_indices]
    sw_heating = sw_heating[source_indices]
    reference_field = reference_field[reference_indices]
    print(f"Analyzing {common_dates[0]} through {common_dates[-1]} ({common_dates.size} days).")

    source_fields: dict[str, np.ndarray] = {}
    if SOURCE_NAME in {"QR", "ALL"}:
        source_fields["QR"] = lw_heating + sw_heating
    if SOURCE_NAME in {"LW", "ALL"}:
        source_fields["LW"] = lw_heating
    if SOURCE_NAME in {"SW", "ALL"}:
        source_fields["SW"] = sw_heating
    return source_fields, reference_field


# ====================================================
# Preprocessing function
# ====================================================

def prepare_components(
    source_fields: dict[str, np.ndarray],
    reference_field: np.ndarray,
    REFERENCE_KEY: str,
    MERIDIONAL_COMPONENT: workflow.MeridionalComponent,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Remove means, derive omega PCs when selected, and select meridional components."""
    source_anomalies: dict[str, np.ndarray] = {name: workflow.remove_time_zonal_mean(field) for name, field in source_fields.items()}

    reference_anomalies: dict[str, np.ndarray]

    if REFERENCE_KEY == "W":
        
        nt, nz, ny, nx = reference_field.shape # assigning fields

        # calculate zonal and temporal anomalies
        omega_anom: np.ndarray = reference_field - np.nanmean(reference_field, axis=(0, -1), keepdims=True)

        # calculate EOFs of the omega anomalies
        w_eofs, w_pcs = preproc.calculate_eof(
            omega_anom.transpose(1, 0, 2, 3).reshape(nz, -1),
            n_components=2,
        )

        reference_anomalies = {
            f"W{i+1}": -w_pcs[i].reshape(nt, ny, nx)
            for i in range(w_pcs.shape[0])
        }
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


# ====================================================
# Spectral analysis function
# ====================================================

def analyze_spectra(
    source_components: dict[str, np.ndarray],
    reference_components: dict[str, np.ndarray],
    WINDOW_SIZE: int,
    OVERLAP_SIZE: int,
    EQUIVALENT_DEPTHS: tuple[float, float],
) -> dict[str, dict[str, tuple[np.ndarray, ...]]]:
    """Window fields, calculate spectra, print coherence means, and prepare plot arrays.

    Results are indexed by reference name, then heating component. Each entry
    contains wavenumbers, positive frequencies, normalized co-spectrum, and MSC.
    """

    # ------------------------------------------------
    # Detrend and Hann-window overlapping time segments
    # ------------------------------------------------
    chunked_fields: dict[str, np.ndarray] = workflow.segment_fields(
        {**source_components, **reference_components},
        window_size=WINDOW_SIZE,
        overlap_size=OVERLAP_SIZE,
    )
    source_chunks: dict[str, np.ndarray] = {name: chunked_fields[name] for name in source_components}
    reference_chunks: np.ndarray = chunked_fields[next(iter(reference_components))]
    print(f"Windowed chunk shape: {reference_chunks.shape}")

    # ------------------------------------------------
    # Calculate spectra, diagnostics, and plotting arrays
    # ------------------------------------------------
    frequencies, zonal_wavenumbers = workflow.spectral_coordinates(
        WINDOW_SIZE, reference_chunks.shape[-1], sampling_interval=1.0,
    )
    plot_data: dict[str, dict[str, tuple[np.ndarray, ...]]] = {}
    for reference_name in reference_components:
        cross_spectrum, source_power, reference_power = workflow.calculate_segment_spectra(
            source_chunks, chunked_fields[reference_name], window_size=WINDOW_SIZE,
        )
        mean_cross, mean_source_power, mean_reference_power = workflow.average_segment_spectra(
            cross_spectrum, source_power, reference_power,
        )
        diagnostics = workflow.calculate_spectral_diagnostics(
            mean_cross, mean_source_power, mean_reference_power,
        )
        plot_data[reference_name] = {}
        for source_name in source_components:
            wave_domain_coherence = workflow.average_wave_domain_coherence(
                zonal_wavenumbers, frequencies, diagnostics["coherence_squared"][source_name],
                equivalent_depths=EQUIVALENT_DEPTHS,
            )
            print(f"\n{source_name} / {reference_name}")
            print(f"{'Domain':<10} {'Mean coherence²':>17} {'Finite bins':>12}")
            for domain, stats in wave_domain_coherence.items():
                print(f"{domain:<10} {stats['mean_coherence_squared']:>17.4f} {stats['n_bins']:>12d}")
            plot_data[reference_name][source_name] = workflow.prepare_plot_data(
                frequencies, zonal_wavenumbers,
                diagnostics["normalized_cospectrum"][source_name],
                diagnostics["coherence_squared"][source_name],
            )
    return plot_data


# ====================================================
# Plotting function
# ====================================================

def plot_spectra(
    plot_data: dict[str, dict[str, tuple[np.ndarray, ...]]],
    OUTPUT_PATH: Path,
    REFERENCE_KEY: str,
    MERIDIONAL_COMPONENT: workflow.MeridionalComponent,
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
    reference_names: tuple[str, ...] = tuple(plot_data)
    source_names: tuple[str, ...] = tuple(plot_data[reference_names[0]])
    for source_name in source_names:
        n_panels: int = len(reference_names)
        figure, axes = plt.subplots(
            1, n_panels, figsize=(12, 5) if n_panels == 2 else (8, 6),
            sharex=True, sharey=True, constrained_layout=True, squeeze=False,
        )
        for axis, reference_name in zip(axes[0], reference_names):
            workflow.plot_cospectrum_with_coherence(
                *plot_data[reference_name][source_name],
                source_name=source_name,
                reference_name=reference_name,
                shading_levels=np.linspace(-1.0, 1.0, 21), #type: ignore
                coherence_levels=COHERENCE_LEVELS,
                equivalent_depths=EQUIVALENT_DEPTHS,
                cmap="RdBu_r",
                extend="both",
                colorbar_label="Normalized co-spectrum",
                axis=axis,
            )
            axis.set_title(f"{source_name} / {reference_name} ({MERIDIONAL_COMPONENT})")
        figure_path: Path = OUTPUT_PATH / f"{source_name}_{REFERENCE_KEY}_{MERIDIONAL_COMPONENT}.png"
        figure.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved {figure_path}")
        if SHOW_FIGURES:
            plt.show()
        plt.close(figure)


# ====================================================
# Main function
# ====================================================

def main(
    DATA_PATH: Path,
    OUTPUT_PATH: Path,
    LATITUDE_BOUNDS: tuple[float, float],
    WINDOW_SIZE: int,
    OVERLAP_SIZE: int,
    MERIDIONAL_COMPONENT: workflow.MeridionalComponent,
    REFERENCE_KEY: str,
    SOURCE_NAME: str = "QR",
    START_DATE: str | None = "2006-01-01",
    END_DATE: str | None = "2018-01-01",
    COHERENCE_LEVELS: tuple[float, ...] | None = None,
    EQUIVALENT_DEPTHS: tuple[float, float] = (8.0, 90.0),
    FIGURE_DPI: int = 600,
    SHOW_FIGURES: bool = False,
) -> None:
    """Run loading, preprocessing, spectral analysis, and plotting in sequence.

    REFERENCE_KEY: "CWV", "precipitation" (or "Prec"), "W", "W1", or "W2".
    "W" (or "omega") plots both negative omega PCs side by side.
    SOURCE_NAME: "QR" (LW + SW), "LW", "SW", or "ALL" for all three figures.
    EOF normalization and PC signs follow Qr_omega.ipynb unchanged.
    Inputs are assumed to have complete daily records on the prepared grids.
    """
    reference_aliases: dict[str, str] = {
        "cwv": "CWV", "prec": "precipitation", "precipitation": "precipitation",
        "omega": "W", "w": "W", "w1": "W1", "w2": "W2",
    }
    REFERENCE_KEY = reference_aliases[REFERENCE_KEY.lower()]
    SOURCE_NAME = SOURCE_NAME.upper()
    if COHERENCE_LEVELS is None:
        COHERENCE_LEVELS = (0.3, 0.4, 0.5, 0.6, 0.8) if REFERENCE_KEY == "CWV" else (0.5, 0.6, 0.8)

    # Load heating and reference fields on common dates.
    source_fields, reference_field = load_analysis_fields(
        DATA_PATH, LATITUDE_BOUNDS, REFERENCE_KEY, SOURCE_NAME, START_DATE, END_DATE,
    )

    # Remove means and select symmetric, antisymmetric, or full fields.
    source_components, reference_components = prepare_components(
        source_fields, reference_field, REFERENCE_KEY, MERIDIONAL_COMPONENT,
    )
    del source_fields, reference_field

    # Calculate normalized co-spectrum and squared coherence.
    plot_data = analyze_spectra(
        source_components, reference_components, WINDOW_SIZE, OVERLAP_SIZE, EQUIVALENT_DEPTHS,
    )

    # Save figures for the selected heating components.
    plot_spectra(
        plot_data, OUTPUT_PATH, REFERENCE_KEY, MERIDIONAL_COMPONENT,
        COHERENCE_LEVELS, EQUIVALENT_DEPTHS, FIGURE_DPI, SHOW_FIGURES,
    )

# ====================================================
# Execute Main function
# ====================================================

if __name__ == "__main__":

    # List of reference key:
    reference_keys = ["CWV", "W"]
    source_keys = ["QR", "LW", "SW"]

    iters: list[tuple[str, str]] = list(product(reference_keys, source_keys))

    for reference_key, source_key in iters:
        main(
                DATA_PATH=Path("/data92/b11209013"),
                OUTPUT_PATH=Path("/home/b11209013/Manuscript_Prep/KW_CRI/Figure/New_analysis_Figure/Spectral_Analysis"),
                LATITUDE_BOUNDS=(-5.0, 5.0),
                WINDOW_SIZE=96,
                OVERLAP_SIZE=48,
                MERIDIONAL_COMPONENT="symmetric",
                REFERENCE_KEY=reference_key,  # CWV, precipitation (or Prec), W (both PCs), W1, W2
                SOURCE_NAME=source_key,  # QR = LW + SW; LW, SW, or ALL
                START_DATE="2006-01-01",  # Inclusive; None uses all available dates
                END_DATE="2018-01-01",  # Exclusive; None uses all available dates
                COHERENCE_LEVELS=None,  # None uses the corresponding notebook's levels
                EQUIVALENT_DEPTHS=(8.0, 90.0),
                FIGURE_DPI=300,
                SHOW_FIGURES=False,
            )
