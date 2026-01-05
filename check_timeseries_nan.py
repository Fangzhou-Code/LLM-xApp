#!/usr/bin/env python3
"""
Usage:
    python3 check_timeseries_nan.py path/to/timeseries_xxx.csv
Counts NaNs for shortfall1/shortfall2 and system_reliability_severity (or fallback).
Also computes mean/std in time window [400,800].
"""
import sys
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Path to timeseries CSV")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    # Column candidates for system reliability severity
    sev_col = None
    for c in ("system_reliability_severity", "system_reliability"):
        if c in df.columns:
            sev_col = c
            break

    cols = []
    for key in ("shortfall1", "shortfall2"):
        if key in df.columns:
            cols.append(key)
        else:
            print(f"WARNING: column {key} not found in {args.csv}")
    if sev_col:
        cols.append(sev_col)
    else:
        print(f"WARNING: no system_reliability_severity/system_reliability column found in {args.csv}")

    print("File:", args.csv)
    print("Total rows:", len(df))
    for c in cols:
        nans = int(df[c].isna().sum())
        print(f"NaN count in '{c}': {nans}")

    # time window stats
    if "t" not in df.columns:
        print("No 't' column found; cannot compute 400..800s window stats.")
        return

    window = df[(df["t"] >= 400) & (df["t"] <= 800)]
    if window.empty:
        print("No rows in t in [400,800].")
        return

    print("\nWindow t in [400,800]: rows =", len(window))
    for c in cols:
        # convert to numeric, ignore NaN for mean/std
        vals = pd.to_numeric(window[c], errors="coerce").dropna().astype(float)
        if vals.empty:
            print(f"{c}: no valid numeric samples in window (all NaN or non-numeric).")
        else:
            print(f"{c}: mean={vals.mean():.6f}, std={vals.std(ddof=0):.6f}, count={len(vals)}")

if __name__ == "__main__":
    main()
