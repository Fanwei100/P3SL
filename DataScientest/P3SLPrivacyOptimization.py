"""Utilities implementing the P3SL split/noise optimizer from the paper.

The paper describes a server-side Privacy Leakage Table PL(s, sigma), a
Noise Assignment Table T_sigma[s], and a client-side finite enumeration over
feasible split points using

    alpha_i * FSIM(s, sigma) + (1 - alpha_i) * E_i^total(s).

This module keeps that logic small and reusable by both the coordinator and
client workers.  It intentionally uses table lookups and finite enumeration,
matching the discrete optimization described in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import math
import pandas as pd

Number = Union[int, float]


_SPLIT_COLUMNS = ("SplitLayer", "Split Layer", "split", "Split", "s", "si")
_ENERGY_COLUMNS = ("Total", "Energy", "Etotal", "E_total", "TotalEnergy")
_PEAK_COLUMNS = (
    "PeakPower",
    "Peak Power",
    "MaxPower",
    "Max Power",
    "p_peak",
    "Ppeak",
    "MaxEnergy",  # legacy column name used by the existing profiler
)


@dataclass(frozen=True)
class SplitSelection:
    """Result returned by the client-side split-point enumeration."""

    split: int
    noise: float
    objective: float
    privacy: float
    energy: float
    peak_power: Optional[float]
    alpha: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "noise": self.noise,
            "objective": self.objective,
            "privacy": self.privacy,
            "energy": self.energy,
            "peak_power": self.peak_power,
            "alpha": self.alpha,
        }


def _coerce_float_key(value: Any) -> float:
    """Convert a CSV/dict key representing sigma to a rounded float."""

    return round(float(value), 10)


def load_privacy_table(path: Union[str, Path]) -> pd.DataFrame:
    """Load PL(s, sigma) from a CSV with a SplitLayer/Split Layer column."""

    df = pd.read_csv(path)
    split_col = next((c for c in _SPLIT_COLUMNS if c in df.columns), df.columns[0])
    df = df.rename(columns={split_col: "SplitLayer"}).set_index("SplitLayer")
    df.index = df.index.astype(int)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    # Keep sigma columns ordered numerically while retaining string labels.
    ordered_cols = sorted(df.columns, key=lambda c: float(c))
    return df[ordered_cols]


def privacy_table_to_payload(table: Union[str, Path, pd.DataFrame, Mapping[Any, Any]]) -> Dict[int, Dict[str, float]]:
    """Convert a privacy table to a JSON-serialisable nested dictionary."""

    df = ensure_privacy_table(table)
    payload: Dict[int, Dict[str, float]] = {}
    for split, row in df.iterrows():
        payload[int(split)] = {str(float(col)): float(value) for col, value in row.items() if pd.notna(value)}
    return payload


def ensure_privacy_table(table: Union[str, Path, pd.DataFrame, Mapping[Any, Any]]) -> pd.DataFrame:
    """Accept CSV path, DataFrame, or JSON-like dict and return PL as DataFrame."""

    if isinstance(table, (str, Path)):
        return load_privacy_table(table)
    if isinstance(table, pd.DataFrame):
        df = table.copy()
        if "SplitLayer" in df.columns or "Split Layer" in df.columns:
            split_col = "SplitLayer" if "SplitLayer" in df.columns else "Split Layer"
            df = df.rename(columns={split_col: "SplitLayer"}).set_index("SplitLayer")
        df.index = df.index.astype(int)
        df = df.apply(pd.to_numeric, errors="coerce")
        ordered_cols = sorted(df.columns, key=lambda c: float(c))
        return df[ordered_cols]

    # JSON-like payload from the coordinator: {split: {sigma: fsim}}
    rows: Dict[int, Dict[str, float]] = {}
    for split, values in table.items():
        if isinstance(values, Mapping):
            rows[int(split)] = {str(float(sigma)): float(score) for sigma, score in values.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "SplitLayer"
    df.index = df.index.astype(int)
    ordered_cols = sorted(df.columns, key=lambda c: float(c))
    return df[ordered_cols].apply(pd.to_numeric, errors="coerce")


def normalise_table(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise all PL entries while preserving split/sigma ordering."""

    numeric = df.apply(pd.to_numeric, errors="coerce")
    min_value = float(numeric.min().min())
    max_value = float(numeric.max().max())
    if math.isclose(max_value, min_value):
        return numeric * 0.0
    return (numeric - min_value) / (max_value - min_value)


def _nearest_split(df: pd.DataFrame, split: int) -> int:
    if split in df.index:
        return int(split)
    return int(min(df.index, key=lambda s: abs(int(s) - int(split))))


def _nearest_sigma_column(df: pd.DataFrame, sigma: Number) -> str:
    sigma = float(sigma)
    return min(df.columns, key=lambda c: abs(float(c) - sigma))


def lookup_privacy(table: Union[pd.DataFrame, Mapping[Any, Any], str, Path], split: int, sigma: Number) -> float:
    """Return PL(s, sigma), using the nearest available split/noise bin if needed."""

    df = ensure_privacy_table(table)
    split = _nearest_split(df, int(split))
    col = _nearest_sigma_column(df, sigma)
    return float(df.loc[split, col])


def build_noise_assignment_table(
    table: Union[str, Path, pd.DataFrame, Mapping[Any, Any]],
    fsim_threshold: float,
    smax: int = 10,
    normalise_before_threshold: bool = False,
) -> Dict[int, float]:
    """Build T_sigma[s] = min sigma with PL(s, sigma) <= threshold.

    The threshold should use the same scale as the table by default.  Set
    ``normalise_before_threshold=True`` only when the threshold is defined on
    the normalised leakage index rather than raw FSIM values.
    """

    df = ensure_privacy_table(table)
    work = normalise_table(df) if normalise_before_threshold else df
    noise_by_split: Dict[int, float] = {}
    noise_columns = sorted(work.columns, key=lambda c: float(c))
    for split in range(1, int(smax) + 1):
        row_split = _nearest_split(work, split)
        chosen = float(noise_columns[-1])
        for col in noise_columns:
            value = float(work.loc[row_split, col])
            if value <= fsim_threshold:
                chosen = float(col)
                break
        noise_by_split[split] = chosen
    return noise_by_split


def update_noise_assignment_table(noise_assignment: Mapping[Any, Number], amin: float, accuracy_t: float) -> Dict[int, float]:
    """Apply the paper's noise reassignment rule.

    sigma_{t+1} = sigma_t * (1 - 2 * (A_min - A_t)) when A_t < A_min.
    """

    if accuracy_t >= amin:
        return {int(k): float(v) for k, v in noise_assignment.items()}
    factor = max(0.0, 1.0 - 2.0 * (float(amin) - float(accuracy_t)))
    return {int(k): max(0.0, float(v) * factor) for k, v in noise_assignment.items()}


def _find_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for col in candidates:
        if col.lower() in lowered:
            return lowered[col.lower()]
    return None


def ensure_energy_table(energy_profile: Union[str, Path, pd.DataFrame, Mapping[str, Any]]) -> pd.DataFrame:
    """Return an energy table with columns split, energy, and optional peak."""

    if isinstance(energy_profile, (str, Path)):
        path = Path(energy_profile)
        if not path.exists():
            raise FileNotFoundError(f"Energy profile not found: {path}")
        raw = pd.read_csv(path)
    elif isinstance(energy_profile, pd.DataFrame):
        raw = energy_profile.copy()
    else:
        raw = pd.DataFrame(energy_profile)

    split_col = _find_column(raw, _SPLIT_COLUMNS)
    if split_col is None:
        raise ValueError(f"Energy profile must include one of {_SPLIT_COLUMNS}; found {list(raw.columns)}")

    energy_col = _find_column(raw, _ENERGY_COLUMNS)
    if energy_col is None:
        if "Communication" in raw.columns and "Computation" in raw.columns:
            raw["Total"] = pd.to_numeric(raw["Communication"], errors="coerce") + pd.to_numeric(raw["Computation"], errors="coerce")
            energy_col = "Total"
        else:
            raise ValueError(f"Energy profile must include Total/Energy or Communication+Computation columns; found {list(raw.columns)}")

    peak_col = _find_column(raw, _PEAK_COLUMNS)
    out = pd.DataFrame(
        {
            "split": pd.to_numeric(raw[split_col], errors="coerce").astype("Int64"),
            "energy": pd.to_numeric(raw[energy_col], errors="coerce"),
        }
    )
    if peak_col is not None:
        out["peak_power"] = pd.to_numeric(raw[peak_col], errors="coerce")
    else:
        out["peak_power"] = pd.NA
    out = out.dropna(subset=["split", "energy"]).copy()
    out["split"] = out["split"].astype(int)
    out = out.sort_values("split").drop_duplicates(subset=["split"], keep="last")
    return out.reset_index(drop=True)


def _normalise_series(series: pd.Series) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if math.isclose(max_value, min_value):
        return series * 0.0
    return (series - min_value) / (max_value - min_value)


def select_split_point(
    privacy_table: Union[str, Path, pd.DataFrame, Mapping[Any, Any]],
    noise_assignment: Mapping[Any, Number],
    energy_profile: Union[str, Path, pd.DataFrame, Mapping[str, Any]],
    smax: int = 10,
    alpha: float = 0.5,
    max_power: Optional[float] = None,
    restrict_after_energy_min: bool = True,
) -> SplitSelection:
    """Client-side finite enumeration for Eq. (3) in the paper.

    ``alpha`` is the personalized privacy sensitivity coefficient alpha_i:
    larger alpha values put more weight on FSIM/privacy leakage, while smaller
    values prioritise energy consumption.
    """

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    energy = ensure_energy_table(energy_profile)
    feasible = energy[energy["split"] <= int(smax)].copy()
    if max_power is not None and "peak_power" in feasible.columns and feasible["peak_power"].notna().any():
        feasible = feasible[feasible["peak_power"].fillna(float("inf")) <= float(max_power)].copy()
    if feasible.empty:
        raise ValueError("No feasible split point remains after applying smax/power constraints")

    feasible = feasible.sort_values("split")
    if restrict_after_energy_min and len(feasible) > 1:
        # Mirrors the manuscript: if energy decreases with depth, start the
        # enumeration at the split point with minimum energy; otherwise keep
        # the full feasible prefix.
        first_energy = float(feasible.iloc[0]["energy"])
        last_energy = float(feasible.iloc[-1]["energy"])
        if last_energy < first_energy:
            min_energy_split = int(feasible.loc[feasible["energy"].idxmin(), "split"])
            feasible = feasible[feasible["split"] >= min_energy_split].copy()

    feasible["energy_norm"] = _normalise_series(feasible["energy"])
    pl_norm = normalise_table(ensure_privacy_table(privacy_table))

    noise_by_split = {int(k): float(v) for k, v in noise_assignment.items()}
    candidates = []
    for _, row in feasible.iterrows():
        split = int(row["split"])
        sigma = noise_by_split.get(split)
        if sigma is None:
            # Fall back to the nearest split present in T_sigma.
            nearest = min(noise_by_split, key=lambda s: abs(s - split))
            sigma = noise_by_split[nearest]
        privacy = lookup_privacy(pl_norm, split, sigma)
        energy_norm = float(row["energy_norm"])
        score = alpha * privacy + (1.0 - alpha) * energy_norm
        peak_value = None if pd.isna(row.get("peak_power", pd.NA)) else float(row["peak_power"])
        candidates.append(SplitSelection(split, float(sigma), float(score), float(privacy), float(row["energy"]), peak_value, alpha))

    return min(candidates, key=lambda c: (c.objective, c.energy, c.split))
