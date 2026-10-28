"""Generate a synthetic, drifted incoming batch for the Churn Prediction Lab 13 test.

Run this script from the project root or place it in the project root and run it
with the project's Python environment. It does not modify data/raw/churn.csv.
The generated records are synthetic test data, not real new customer observations.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import string

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
BASELINE_PATH = RAW_DIR / "churn.csv"
INCOMING_PATH = RAW_DIR / "new_churn.csv"
BACKUP_DIR = ROOT / "test_backups"
BATCH_SIZE = 100
SEED = 13013


def find_column(columns, candidates):
    lookup = {str(column).strip().casefold(): column for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def make_unique_customer_ids(existing_ids, count, rng):
    existing = {str(value).strip() for value in existing_ids if pd.notna(value)}
    created = []
    alphabet = np.array(list(string.ascii_uppercase))
    while len(created) < count:
        prefix = f"{int(rng.integers(0, 10000)):04d}"
        suffix = "".join(rng.choice(alphabet, size=5).tolist())
        candidate = f"{prefix}-{suffix}"
        if candidate not in existing:
            existing.add(candidate)
            created.append(candidate)
    return created


def main():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Main dataset not found: {BASELINE_PATH}")

    baseline = pd.read_csv(BASELINE_PATH)
    if baseline.empty:
        raise ValueError("data/raw/churn.csv is empty.")

    id_col = find_column(baseline.columns, ["customerID", "customer_id", "customer id"])
    monthly_col = find_column(baseline.columns, ["MonthlyCharges", "monthly_charges"])
    total_col = find_column(baseline.columns, ["TotalCharges", "total_charges"])
    tenure_col = find_column(baseline.columns, ["tenure", "Tenure"])

    if not id_col:
        raise ValueError(f"Could not identify the customer ID column. Columns: {list(baseline.columns)}")
    if not monthly_col or not total_col:
        raise ValueError("Required drift columns MonthlyCharges and TotalCharges were not found.")

    rng = np.random.default_rng(SEED)
    monthly = pd.to_numeric(baseline[monthly_col], errors="coerce")
    valid = monthly.notna()
    if valid.sum() < 1:
        raise ValueError(f"No numeric values found in {monthly_col}.")

    # Sample customers from the upper part of the existing monthly-charge
    # distribution. This intentionally shifts the incoming batch upward.
    monthly_cutoff = monthly[valid].quantile(0.60)
    eligible = valid & (monthly >= monthly_cutoff)

    # Also favor longer-tenure customers where that column exists, increasing
    # the chance that TotalCharges shifts upward as well.
    if tenure_col:
        tenure = pd.to_numeric(baseline[tenure_col], errors="coerce")
        valid_tenure = tenure.notna()
        if valid_tenure.any():
            tenure_cutoff = tenure[valid_tenure].quantile(0.50)
            eligible = eligible & (tenure >= tenure_cutoff)

    candidates = baseline.loc[eligible]
    if candidates.empty:
        candidates = baseline.loc[valid]
    sampled = candidates.sample(n=BATCH_SIZE, replace=len(candidates) < BATCH_SIZE, random_state=SEED).copy()
    batch = sampled.reset_index(drop=True)

    # Create never-before-seen IDs while retaining the standard 4-digits-5-letters
    # Telco-style shape. ID uniqueness is checked against the entire main dataset.
    batch[id_col] = make_unique_customer_ids(baseline[id_col], len(batch), rng)

    # Shift MonthlyCharges upward while keeping the values in a plausible range.
    selected_monthly = pd.to_numeric(batch[monthly_col], errors="coerce")
    baseline_max = float(monthly[valid].max())
    shifted_monthly = selected_monthly + rng.uniform(12.0, 22.0, size=len(batch))
    batch[monthly_col] = shifted_monthly.clip(lower=0, upper=baseline_max + 22.0).round(2)

    # Derive higher accumulated charges from MonthlyCharges and tenure. For
    # rows without a usable tenure value, increase their original TotalCharges.
    original_total = pd.to_numeric(batch[total_col], errors="coerce")
    if tenure_col:
        batch_tenure = pd.to_numeric(batch[tenure_col], errors="coerce")
        derived_total = pd.to_numeric(batch[monthly_col], errors="coerce") * batch_tenure.clip(lower=0)
        derived_total = derived_total * rng.uniform(0.90, 1.08, size=len(batch))
        usable = batch_tenure.notna()
        shifted_total = original_total * 1.30 + pd.to_numeric(batch[monthly_col], errors="coerce") * 2.0
        shifted_total.loc[usable] = derived_total.loc[usable]
    else:
        shifted_total = original_total * 1.30 + pd.to_numeric(batch[monthly_col], errors="coerce") * 2.0

    # Preserve missing values as missing and round numeric charge values.
    batch[total_col] = shifted_total.round(2)

    # Back up the previous incoming batch before replacing it.
    if INCOMING_PATH.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"new_churn_before_drift_{stamp}.csv"
        pd.read_csv(INCOMING_PATH).to_csv(backup_path, index=False)
        print(f"[BACKUP] Previous incoming batch saved to: {backup_path.relative_to(ROOT)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    batch.to_csv(INCOMING_PATH, index=False)

    # Verify the batch can be read back and does not overlap with the main IDs.
    written = pd.read_csv(INCOMING_PATH)
    baseline_ids = {str(value).strip() for value in baseline[id_col].dropna()}
    new_ids = [str(value).strip() for value in written[id_col].dropna()]
    overlap = baseline_ids.intersection(new_ids)
    duplicate_new_ids = len(new_ids) - len(set(new_ids))
    if len(written) != BATCH_SIZE:
        raise RuntimeError(f"Expected {BATCH_SIZE} rows but wrote {len(written)}.")
    if overlap or duplicate_new_ids:
        raise RuntimeError(f"ID verification failed: baseline overlap={len(overlap)}, duplicate new IDs={duplicate_new_ids}.")

    base_monthly_mean = monthly.mean()
    incoming_monthly_mean = pd.to_numeric(written[monthly_col], errors="coerce").mean()
    base_total_mean = pd.to_numeric(baseline[total_col], errors="coerce").mean()
    incoming_total_mean = pd.to_numeric(written[total_col], errors="coerce").mean()

    print("\n[SUCCESS] Synthetic drifted Lab 13 batch generated.")
    print(f"Main dataset rows       : {len(baseline)}")
    print(f"Incoming batch rows    : {len(written)}")
    print(f"Customer ID column     : {id_col}")
    print(f"ID overlap with main   : {len(overlap)}")
    print(f"Duplicate IDs in batch : {duplicate_new_ids}")
    print(f"Columns preserved      : {list(written.columns) == list(baseline.columns)}")
    print(f"Incoming file          : {INCOMING_PATH.relative_to(ROOT)}")
    print(f"MonthlyCharges mean    : {base_monthly_mean:.2f} -> {incoming_monthly_mean:.2f}")
    print(f"TotalCharges mean      : {base_total_mean:.2f} -> {incoming_total_mean:.2f}")
    print("\nNOTE: This is synthetic test data designed to shift charge distributions.")
    print("It does not guarantee promotion; the candidate may fail quality/lifecycle gates.")


if __name__ == "__main__":
    main()
