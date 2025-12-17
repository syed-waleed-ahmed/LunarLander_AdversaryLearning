from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def read_sb3_monitor_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Reads a Stable-Baselines3 monitor.csv file.

    Format:
      line 1: #{...json...}
      line 2: r,l,t
      next : values
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Read metadata (first line)
    with csv_path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    meta = {}
    if first.startswith("#"):
        try:
            meta = json.loads(first[1:])
        except json.JSONDecodeError:
            meta = {}

    # Read the actual CSV (skip metadata row)
    df = pd.read_csv(csv_path, skiprows=1)

    # Standardize column names
    df = df.rename(columns={"r": "return", "l": "length", "t": "wall_time_s"})
    df["csv_path"] = str(csv_path)
    df["run_name"] = csv_path.parent.name  # e.g., PPO_Protagonist_0
    df["env_id"] = meta.get("env_id", None)
    df["t_start"] = meta.get("t_start", None)

    # Add useful derived columns
    df["episode"] = range(1, len(df) + 1)
    df["timesteps"] = df["length"].cumsum()

    return df


def collect_monitor_dfs(log_root: str | Path) -> pd.DataFrame:
    """
    Collects ALL monitor.csv under log_root (including nested folders).
    Works even if you later add logs/PPO_Latent_Seed0/monitor.csv, etc.
    """
    log_root = Path(log_root)
    csvs = list(log_root.rglob("monitor.csv"))
    if not csvs:
        raise FileNotFoundError(f"No monitor.csv found under: {log_root}")

    dfs = [read_sb3_monitor_csv(p) for p in csvs]
    return pd.concat(dfs, ignore_index=True)
