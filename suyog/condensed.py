"""
=============================================================================
condensed.py  –  Sliding-Window Feature Extraction for Fault Classification
=============================================================================
Reads   : suyog/dataset.xlsx   (raw simulation signals + wavelet coefficients)
Writes  : suyog/condensed_dataset.xlsx

Processing per window (20 rows):
  Neurons 1-3  : Positive / Negative / Zero sequence magnitudes for CURRENT
  Neurons 4-6  : Positive / Negative / Zero sequence magnitudes for VOLTAGE
  Neurons 7-12 : Sum-of-Squares (energy) of D1 wavelet coefficients
                 for branches  ia, ia1, ia2, ia3, va, va1
All D2, D3 … D10 columns are dropped.
=============================================================================
"""

import os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE    = os.path.join(SCRIPT_DIR, "dataset.xlsx")
OUTPUT_FILE   = os.path.join(SCRIPT_DIR, "condensed_dataset.xlsx")
WINDOW_SIZE   = 20        # sliding-window length (rows)
WINDOW_STEP   = 20        # non-overlapping windows (change to < 20 for overlap)

# Phase signal columns used for sequence-component calculation
CURRENT_PHASES = ["ia", "ib", "ic"]
VOLTAGE_PHASES = ["va", "vb", "vc"]

# D1 wavelet columns whose energy (sum of squares) fills Neurons 7–12
D1_ENERGY_COLS = ["D1_ia", "D1_ia1", "D1_ia2", "D1_ia3", "D1_va", "D1_va1"]

# Fortescue operator  a = e^{j2π/3}
A = np.exp(1j * 2 * np.pi / 3)

# Fortescue transformation matrix  (1/3) · [[1, 1, 1], [1, a, a²], [1, a², a]]
FORTESCUE = (1.0 / 3.0) * np.array([
    [1,    1,     1    ],   # Zero  sequence
    [1,    A,     A**2 ],   # Positive sequence
    [1,    A**2,  A    ],   # Negative sequence
])


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def sequence_magnitudes(phase_a: np.ndarray,
                        phase_b: np.ndarray,
                        phase_c: np.ndarray) -> tuple:
    """
    Compute Positive, Negative, and Zero sequence magnitudes from three
    phase signals over a window.

    Steps
    -----
    1.  Take the mean of each phase within the window to get a representative
        quasi-phasor value.
    2.  Apply the Fortescue transformation.
    3.  Return the magnitudes of [Positive, Negative, Zero].
    """
    # Representative value for each phase in this window
    phasors = np.array([phase_a.mean(), phase_b.mean(), phase_c.mean()])

    # Fortescue: [V0, V1, V2]
    seq = FORTESCUE @ phasors

    mag_zero     = np.abs(seq[0])
    mag_positive = np.abs(seq[1])
    mag_negative = np.abs(seq[2])

    return mag_positive, mag_negative, mag_zero


def energy(signal_window: np.ndarray) -> float:
    """Sum of squares (energy) of a signal within the window."""
    return float(np.sum(signal_window ** 2))


def find_label_column(df: pd.DataFrame) -> str | None:
    """Auto-detect the label / fault-type column."""
    candidates = [
        "Fault_Type", "fault_type", "FaultType",
        "label", "Label", "class", "Class", "fault", "Fault",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"[INFO] Reading {INPUT_FILE} …")
    df = pd.read_excel(INPUT_FILE)
    print(f"[INFO] Raw dataset shape : {df.shape}")
    print(f"[INFO] Columns           : {list(df.columns)}\n")

    # ── 2. Identify & separate label column (if present) ─────────────────────
    label_col = find_label_column(df)
    if label_col:
        print(f"[INFO] Label column detected: '{label_col}'")
    else:
        print("[WARN] No label column found – output will have no labels.")

    # ── 3. Validate required columns exist ───────────────────────────────────
    required = CURRENT_PHASES + VOLTAGE_PHASES + D1_ENERGY_COLS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following required columns are missing from the dataset:\n"
            f"  {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ── 4. Drop D2 … D10 columns (keep D1 only) ─────────────────────────────
    drop_cols = [c for c in df.columns
                 if any(c.startswith(f"D{i}_") for i in range(2, 11))]
    if drop_cols:
        print(f"[INFO] Dropping {len(drop_cols)} higher-order wavelet columns "
              f"(D2–D10):\n       {drop_cols}\n")
        df.drop(columns=drop_cols, inplace=True)

    # ── 5. Sliding-window feature extraction ─────────────────────────────────
    n_rows   = len(df)
    n_windows = (n_rows - WINDOW_SIZE) // WINDOW_STEP + 1
    print(f"[INFO] Total rows      : {n_rows}")
    print(f"[INFO] Window size     : {WINDOW_SIZE}")
    print(f"[INFO] Window step     : {WINDOW_STEP}")
    print(f"[INFO] Windows to emit : {n_windows}\n")

    records = []

    for w in range(n_windows):
        start = w * WINDOW_STEP
        end   = start + WINDOW_SIZE
        win   = df.iloc[start:end]

        # ---- Neurons 1–3 : Current sequence magnitudes ----
        I_pos, I_neg, I_zero = sequence_magnitudes(
            win[CURRENT_PHASES[0]].values,
            win[CURRENT_PHASES[1]].values,
            win[CURRENT_PHASES[2]].values,
        )

        # ---- Neurons 4–6 : Voltage sequence magnitudes ----
        V_pos, V_neg, V_zero = sequence_magnitudes(
            win[VOLTAGE_PHASES[0]].values,
            win[VOLTAGE_PHASES[1]].values,
            win[VOLTAGE_PHASES[2]].values,
        )

        # ---- Neurons 7–12 : D1 wavelet energy ----
        d1_energies = [energy(win[col].values) for col in D1_ENERGY_COLS]

        # ---- Assemble row ----
        row = {
            "I_Positive_Seq": I_pos,
            "I_Negative_Seq": I_neg,
            "I_Zero_Seq":     I_zero,
            "V_Positive_Seq": V_pos,
            "V_Negative_Seq": V_neg,
            "V_Zero_Seq":     V_zero,
            "Energy_D1_ia":   d1_energies[0],
            "Energy_D1_ia1":  d1_energies[1],
            "Energy_D1_ia2":  d1_energies[2],
            "Energy_D1_ia3":  d1_energies[3],
            "Energy_D1_va":   d1_energies[4],
            "Energy_D1_va1":  d1_energies[5],
        }

        # ---- Carry label (majority vote inside the window) ----
        if label_col:
            row[label_col] = win[label_col].mode().iloc[0]

        records.append(row)

    # ── 6. Build output DataFrame ────────────────────────────────────────────
    out_df = pd.DataFrame(records)

    # Reorder so label is the last column
    if label_col and label_col in out_df.columns:
        cols = [c for c in out_df.columns if c != label_col] + [label_col]
        out_df = out_df[cols]

    print(f"[INFO] Condensed dataset shape : {out_df.shape}")
    print(out_df.head(10).to_string(index=False))
    print()

    # ── 7. Save ──────────────────────────────────────────────────────────────
    out_df.to_excel(OUTPUT_FILE, index=False)
    print(f"[✔] Saved → {OUTPUT_FILE}")
    print("[DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
