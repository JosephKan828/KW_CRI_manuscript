#!/usr/bin/env python3
"""Convert ERA5 pressure vertical velocity (omega) to geometric velocity.

Dry-air density is calculated from the ideal gas law,

    rho = p / (R_d T),

and geometric vertical velocity is then calculated from hydrostatic balance,

    w = -omega / (rho g).

Positive ``w`` denotes upward motion.  The input files used by this project are
large, so data are read and written in small time chunks instead of being loaded
entirely into memory.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np


DEFAULT_OMEGA = Path(
    "/data92/b11209013/ERA5_GRIB/Data/tropical_-10_10/W_no_rm3harm.nc"
)
DEFAULT_TEMPERATURE = Path(
    "/data92/b11209013/ERA5_GRIB/Data/tropical_-10_10/T_no_rm3harm.nc"
)
DEFAULT_OUTPUT = Path(
    "/data92/b11209013/ERA5_GRIB/Data/tropical_-10_10/w_no_rm3harm.nc"
)

R_D = 287.05  # J kg-1 K-1, dry-air gas constant
G = 9.80665  # m s-2, standard gravitational acceleration
FILL_VALUE = np.float32(9.96921e36)
COORDINATES = ("time", "plev", "lat", "lon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--temperature", type=Path, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=1,
        help="Number of time steps processed together (default: 1).",
    )
    parser.add_argument(
        "--include-density",
        action="store_true",
        help="Also store air density in the output (substantially increases file size).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    return parser.parse_args()


def copy_attributes(source: nc.Variable, target: nc.Variable) -> None:
    """Copy variable attributes except the creation-time _FillValue attribute."""
    attributes = {
        name: source.getncattr(name)
        for name in source.ncattrs()
        if name != "_FillValue"
    }
    if attributes:
        target.setncatts(attributes)


def validate_inputs(omega_ds: nc.Dataset, temperature_ds: nc.Dataset) -> None:
    """Fail early if variables, units, dimensions, or coordinates do not align."""
    if "W" not in omega_ds.variables:
        raise KeyError("Omega file does not contain the expected variable 'W'.")
    if "T" not in temperature_ds.variables:
        raise KeyError("Temperature file does not contain the expected variable 'T'.")

    omega = omega_ds.variables["W"]
    temperature = temperature_ds.variables["T"]
    if omega.dimensions != COORDINATES or temperature.dimensions != COORDINATES:
        raise ValueError(
            f"W and T must both have dimensions {COORDINATES}; got "
            f"{omega.dimensions} and {temperature.dimensions}."
        )
    if omega.shape != temperature.shape:
        raise ValueError(f"W and T shapes differ: {omega.shape} versus {temperature.shape}.")

    omega_units = str(getattr(omega, "units", "")).replace(" ", "").lower()
    temperature_units = str(getattr(temperature, "units", "")).strip().lower()
    pressure_units = str(getattr(omega_ds.variables["plev"], "units", "")).strip().lower()
    if omega_units not in {"pas**-1", "pas-1", "pa/s"}:
        raise ValueError(f"Expected W in Pa s-1, found {getattr(omega, 'units', None)!r}.")
    if temperature_units not in {"k", "kelvin"}:
        raise ValueError(f"Expected T in kelvin, found {getattr(temperature, 'units', None)!r}.")
    if pressure_units not in {"pa", "pascal", "pascals"}:
        raise ValueError(f"Expected plev in Pa, found {pressure_units!r}.")

    for coordinate in COORDINATES:
        left = omega_ds.variables[coordinate][:]
        right = temperature_ds.variables[coordinate][:]
        if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1e-10):
            raise ValueError(f"Coordinate {coordinate!r} differs between the input files.")


def create_output(
    output_path: Path,
    omega_ds: nc.Dataset,
    include_density: bool,
) -> nc.Dataset:
    """Create an output dataset and copy coordinate variables and metadata."""
    output = nc.Dataset(output_path, "w", format="NETCDF4")
    for name, dimension in omega_ds.dimensions.items():
        output.createDimension(name, None if dimension.isunlimited() else len(dimension))

    for name in COORDINATES:
        source = omega_ds.variables[name]
        target = output.createVariable(name, source.dtype, source.dimensions)
        copy_attributes(source, target)
        target[:] = source[:]

    chunks = (1, 1, len(omega_ds.dimensions["lat"]), len(omega_ds.dimensions["lon"]))
    w = output.createVariable(
        "w",
        "f4",
        COORDINATES,
        fill_value=FILL_VALUE,
        zlib=True,
        complevel=2,
        shuffle=True,
        chunksizes=chunks,
    )
    w.setncatts(
        {
            "standard_name": "upward_air_velocity",
            "long_name": "Geometric vertical velocity",
            "units": "m s-1",
            "positive": "up",
            "formula": "w = -omega / (rho * g); rho = p / (R_d * T)",
        }
    )

    if include_density:
        rho = output.createVariable(
            "rho",
            "f4",
            COORDINATES,
            fill_value=FILL_VALUE,
            zlib=True,
            complevel=2,
            shuffle=True,
            chunksizes=chunks,
        )
        rho.setncatts(
            {
                "standard_name": "air_density",
                "long_name": "Dry-air density calculated from pressure and temperature",
                "units": "kg m-3",
                "formula": "rho = p / (R_d * T)",
            }
        )

    for attribute in omega_ds.ncattrs():
        if attribute != "history":
            output.setncattr(attribute, omega_ds.getncattr(attribute))
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    old_history = getattr(omega_ds, "history", "")
    new_history = f"{timestamp}: converted omega to geometric vertical velocity"
    output.history = f"{new_history}\n{old_history}" if old_history else new_history
    output.source_omega_file = str(Path(omega_ds.filepath()).resolve())
    output.source_temperature_file = "See conversion command"
    output.dry_air_gas_constant = f"{R_D} J kg-1 K-1"
    output.standard_gravity = f"{G} m s-2"
    return output


def convert(args: argparse.Namespace) -> None:
    if args.time_chunk < 1:
        raise ValueError("--time-chunk must be at least 1.")
    for path in (args.omega, args.temperature):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    if partial_path.exists():
        raise FileExistsError(
            f"Partial output exists: {partial_path}. Remove it before trying again."
        )

    try:
        with nc.Dataset(args.omega, "r") as omega_ds, nc.Dataset(
            args.temperature, "r"
        ) as temperature_ds:
            validate_inputs(omega_ds, temperature_ds)
            ntime = len(omega_ds.dimensions["time"])
            pressure = np.asarray(omega_ds.variables["plev"][:], dtype=np.float32)[
                None, :, None, None
            ]

            with create_output(partial_path, omega_ds, args.include_density) as output:
                output.source_temperature_file = str(args.temperature.expanduser().resolve())
                for start in range(0, ntime, args.time_chunk):
                    stop = min(start + args.time_chunk, ntime)
                    section = np.s_[start:stop, :, :, :]
                    omega = np.ma.masked_invalid(omega_ds.variables["W"][section])
                    temperature = np.ma.masked_invalid(
                        temperature_ds.variables["T"][section]
                    )
                    temperature = np.ma.masked_less_equal(temperature, 0.0)

                    density = pressure / (R_D * temperature)
                    vertical_velocity = -omega / (density * G)
                    output.variables["w"][section] = vertical_velocity.astype(np.float32)
                    if args.include_density:
                        output.variables["rho"][section] = density.astype(np.float32)

                    print(f"Processed time steps {start + 1}-{stop} of {ntime}", flush=True)

        os.replace(partial_path, output_path)
        print(f"Wrote {output_path}")
    except BaseException:
        # Keep no misleading, apparently complete NetCDF file after an interrupted run.
        if partial_path.exists():
            partial_path.unlink()
        raise


if __name__ == "__main__":
    convert(parse_args())
