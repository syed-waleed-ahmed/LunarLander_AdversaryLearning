from __future__ import annotations

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from load_monitor import collect_monitor_dfs


def moving_average(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="logs")
    ap.add_argument("--out", type=str, default="analysis/outputs")
    ap.add_argument("--smooth", type=int, default=20, help="moving average window in episodes")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_monitor_dfs(args.logdir)

    # Plot return vs timesteps
    plt.figure()
    for run_name, g in df.groupby("run_name"):
        g = g.sort_values("timesteps")
        y = moving_average(g["return"], args.smooth)
        plt.plot(g["timesteps"], y, label=run_name)

    plt.xlabel("Timesteps")
    plt.ylabel(f"Episode Return (moving avg window={args.smooth})")
    plt.title("Training Curve (SB3 Monitor)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve_return_vs_timesteps.png", dpi=200)
    plt.close()

    # Plot raw episode return vs episode index (sometimes easier to read)
    plt.figure()
    for run_name, g in df.groupby("run_name"):
        g = g.sort_values("episode")
        y = moving_average(g["return"], args.smooth)
        plt.plot(g["episode"], y, label=run_name)

    plt.xlabel("Episode")
    plt.ylabel(f"Episode Return (moving avg window={args.smooth})")
    plt.title("Training Curve (by Episode)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve_return_vs_episode.png", dpi=200)
    plt.close()

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
