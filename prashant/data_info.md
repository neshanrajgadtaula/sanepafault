# Dataset Feature Documentation
## Power System Fault Classification using Sequence Components and Wavelet Energy Features

---

# 1. Windowing Metadata

| Column Name | Description | Type | Notes |
|---|---|---|---|
| `Window_No` | Unique identifier for each signal window/segment | Integer | Used to track segmented signal blocks |
| `Start_SNo` | Starting sample index of the signal window | Integer | Indicates where the window begins in the original signal |
| `End_SNo` | Ending sample index of the signal window | Integer | Indicates where the window ends in the original signal |

---

# 2. Phase Current Wavelet Energy Features

These features represent the wavelet detail coefficient energy extracted from the three-phase current signals.

---

## Phase-A Current (`ia`)

| Column Name | Description |
|---|---|
| `ia_D1_Energy` | Energy of level-1 wavelet detail coefficients of phase-A current |
| `ia_D2_Energy` | Energy of level-2 wavelet detail coefficients of phase-A current |
| `ia_D3_Energy` | Energy of level-3 wavelet detail coefficients of phase-A current |
| `ia_D4_Energy` | Energy of level-4 wavelet detail coefficients of phase-A current |

---

## Phase-B Current (`ib`)

| Column Name | Description |
|---|---|
| `ib_D1_Energy` | Energy of level-1 wavelet detail coefficients of phase-B current |
| `ib_D2_Energy` | Energy of level-2 wavelet detail coefficients of phase-B current |
| `ib_D3_Energy` | Energy of level-3 wavelet detail coefficients of phase-B current |
| `ib_D4_Energy` | Energy of level-4 wavelet detail coefficients of phase-B current |

---

## Phase-C Current (`ic`)

| Column Name | Description |
|---|---|
| `ic_D1_Energy` | Energy of level-1 wavelet detail coefficients of phase-C current |
| `ic_D2_Energy` | Energy of level-2 wavelet detail coefficients of phase-C current |
| `ic_D3_Energy` | Energy of level-3 wavelet detail coefficients of phase-C current |
| `ic_D4_Energy` | Energy of level-4 wavelet detail coefficients of phase-C current |

---

# 3. Sequence Current Components

Sequence components are derived using symmetrical component transformation.

| Symbol | Meaning |
|---|---|
| `io` | Zero-sequence current |
| `i1` | Positive-sequence current |
| `i2` | Negative-sequence current |

---

## Zero-Sequence Current (`io`)

Useful for detecting:
- Ground faults
- Earth leakage
- Neutral current imbalance

| Column Name | Description |
|---|---|
| `io_D1_Energy` | Energy of level-1 wavelet detail coefficients of zero-sequence current |
| `io_D2_Energy` | Energy of level-2 wavelet detail coefficients of zero-sequence current |
| `io_D3_Energy` | Energy of level-3 wavelet detail coefficients of zero-sequence current |
| `io_D4_Energy` | Energy of level-4 wavelet detail coefficients of zero-sequence current |

---

## Positive-Sequence Current (`i1`)

Represents the balanced component of the system.

| Column Name | Description |
|---|---|
| `i1_D1_Energy` | Energy of level-1 wavelet detail coefficients of positive-sequence current |
| `i1_D2_Energy` | Energy of level-2 wavelet detail coefficients of positive-sequence current |
| `i1_D3_Energy` | Energy of level-3 wavelet detail coefficients of positive-sequence current |
| `i1_D4_Energy` | Energy of level-4 wavelet detail coefficients of positive-sequence current |

---

## Negative-Sequence Current (`i2`)

Useful for detecting:
- Unbalanced faults
- Line-to-line faults
- Phase asymmetry

| Column Name | Description |
|---|---|
| `i2_D1_Energy` | Energy of level-1 wavelet detail coefficients of negative-sequence current |
| `i2_D2_Energy` | Energy of level-2 wavelet detail coefficients of negative-sequence current |
| `i2_D3_Energy` | Energy of level-3 wavelet detail coefficients of negative-sequence current |
| `i2_D4_Energy` | Energy of level-4 wavelet detail coefficients of negative-sequence current |

---

# 4. Phase Voltage Wavelet Energy Features

These features represent wavelet energy extracted from three-phase voltage signals.

---

## Phase-A Voltage (`va`)

| Column Name | Description |
|---|---|
| `va_D1_Energy` | Energy of level-1 wavelet detail coefficients of phase-A voltage |
| `va_D2_Energy` | Energy of level-2 wavelet detail coefficients of phase-A voltage |
| `va_D3_Energy` | Energy of level-3 wavelet detail coefficients of phase-A voltage |
| `va_D4_Energy` | Energy of level-4 wavelet detail coefficients of phase-A voltage |

---

## Phase-B Voltage (`vb`)

| Column Name | Description |
|---|---|
| `vb_D1_Energy` | Energy of level-1 wavelet detail coefficients of phase-B voltage |
| `vb_D2_Energy` | Energy of level-2 wavelet detail coefficients of phase-B voltage |
| `vb_D3_Energy` | Energy of level-3 wavelet detail coefficients of phase-B voltage |
| `vb_D4_Energy` | Energy of level-4 wavelet detail coefficients of phase-B voltage |

---

## Phase-C Voltage (`vc`)

| Column Name | Description |
|---|---|
| `vc_D1_Energy` | Energy of level-1 wavelet detail coefficients of phase-C voltage |
| `vc_D2_Energy` | Energy of level-2 wavelet detail coefficients of phase-C voltage |
| `vc_D3_Energy` | Energy of level-3 wavelet detail coefficients of phase-C voltage |
| `vc_D4_Energy` | Energy of level-4 wavelet detail coefficients of phase-C voltage |

---

# 5. Sequence Voltage Components

| Symbol | Meaning |
|---|---|
| `vo` | Zero-sequence voltage |
| `v1` | Positive-sequence voltage |
| `v2` | Negative-sequence voltage |

---

## Zero-Sequence Voltage (`vo`)

Useful for:
- Ground fault detection
- Neutral displacement analysis

| Column Name | Description |
|---|---|
| `vo_D1_Energy` | Energy of level-1 wavelet detail coefficients of zero-sequence voltage |
| `vo_D2_Energy` | Energy of level-2 wavelet detail coefficients of zero-sequence voltage |
| `vo_D3_Energy` | Energy of level-3 wavelet detail coefficients of zero-sequence voltage |
| `vo_D4_Energy` | Energy of level-4 wavelet detail coefficients of zero-sequence voltage |

---

## Positive-Sequence Voltage (`v1`)

Represents the healthy balanced voltage component.

| Column Name | Description |
|---|---|
| `v1_D1_Energy` | Energy of level-1 wavelet detail coefficients of positive-sequence voltage |
| `v1_D2_Energy` | Energy of level-2 wavelet detail coefficients of positive-sequence voltage |
| `v1_D3_Energy` | Energy of level-3 wavelet detail coefficients of positive-sequence voltage |
| `v1_D4_Energy` | Energy of level-4 wavelet detail coefficients of positive-sequence voltage |

---

## Negative-Sequence Voltage (`v2`)

Useful for:
- Voltage imbalance analysis
- Fault asymmetry detection

| Column Name | Description |
|---|---|
| `v2_D1_Energy` | Energy of level-1 wavelet detail coefficients of negative-sequence voltage |
| `v2_D2_Energy` | Energy of level-2 wavelet detail coefficients of negative-sequence voltage |
| `v2_D3_Energy` | Energy of level-3 wavelet detail coefficients of negative-sequence voltage |
| `v2_D4_Energy` | Energy of level-4 wavelet detail coefficients of negative-sequence voltage |

---

# 6. Target Label

| Column Name | Description |
|---|---|
| `Fault_Label` | Class label representing the fault condition associated with the signal window |

---

# 7. Wavelet Decomposition Notes

| Term | Meaning |
|---|---|
| `D1` | Highest-frequency wavelet detail band |
| `D2` | High-frequency detail band |
| `D3` | Medium-frequency detail band |
| `D4` | Lower-frequency detail band |

---

# 8. Energy Feature Definition

Wavelet energy is typically computed as:

Energy = Σ(detail_coefficients²)

Higher energy values generally indicate:
- Fault transients
- Abrupt disturbances
- Harmonic content
- System imbalance

---

# 9. Dataset Purpose

This dataset is intended for:
- Power system fault detection
- Fault classification
- Protection relay research
- Smart grid analytics
- Machine learning based disturbance analysis

---