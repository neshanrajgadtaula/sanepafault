# SANEPA FAULT
Added Load Flow only on Kandevtasthan opposite
![alt text](<Screenshot 2026-04-27 at 5.57.52 pm.png>)
Added Load Flow upto Pulchowk Campus
![alt text](<Screenshot 2026-04-28 at 10.45.43 pm.png>)
Slight drop in voltage is seen as we keep adding the load.
## After completion of all the loads
![alt text](<Screenshot 2026-04-29 at 2.26.54 pm-1.png>)

The load graph is shown as: 
![alt text](load_voltages_plot-1.png)
Blue line — voltage trend connecting the loads

Green dots — loads at normal voltage (≥ 0.98 pu)

Orange dots — loads with low voltage (below 0.98 pu) — Jal Utpanna Office, Pulchowk Campus, and NTC Office Loads

Green dashed line — nominal voltage (1.0 pu)

Orange dotted line — warning threshold (0.98 pu)

Annotations — max (High Court Load: 0.9953) and min (Pulchowk Campus Load: 0.9750)

A red category for critical voltages (outside ±5%) is defined in the legend in case future runs of load flow produce values that severe.

## Branch Priortization

| # | Sub-feeder name (by load cluster) | Approx. load served | Distribution transformers on it | Comments |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Kandevatasthan / Bima Samiti lateral (west-southwest) | ~165 kW | Kandebata Sthan 300 kVA, Bima Samiti 200 kVA, Sarbanga Hospital 200 kVA | Long lateral going southwest  has the largest single load (Kandevatasthan, 68.8 kW) |
| 2 | Pulchowk / Bakhundol lateral (central-south) | ~245 kW | Pulchowk Campus Wall 100 kVA, Bakhundol Mode 300 kVA, Bakhundol Car Wash 200 kVA, Bakhundol Kutumba 200 kVA | Densest cluster includes the second largest LV transformer (300 kVA Bakhundol Mode) and contains 4 transformers |
| 3 | Sarwanga Hospital / Pariwar Niyojan / High Court lateral (south) | ~122 kW | Sarwanga Hospital 300 kVA, Pariwar Niyojan 100 kVA, High Court 200 kVA | Critical loads (hospital + court)  strong candidate for solar siting and protection priority |
| 4 | NTC / Sajha / NEA / Manab Adhikar / Labim corridor (east, the longest path) | ~250 kW | NTC Office, Sajha 500 kVA, NEA Charging Station 200 kVA, three more 200 kVA transformers, Norvic Hospital, Labim Mall | The longest electrical distance from source voltage drop is worst here, includes the biggest single transformer (Sajha 500 kVA) |
| 5 | Jal Utpanna lateral (southeast) | ~70 kW | Jal Utpanna 100 kVA + wallside load | Short lateral off the eastern corridor |





# Update by Santosh

## Project Update

The updated simulation model has been pushed to GitHub at:

/Santosh/Sanepa_overall_model_Fault.slx

The complete dataset, including the extracted wavelet features, has also been added at:

/Santosh/dataset.xlsx


## Simulation Details

The simulation was performed for a total time of 5 seconds.

The fault was applied from 1 second to 2 seconds.

The fault type used in this case was ABCG fault.

## Signals Used

The updated model includes current and voltage signals such as:

ia, ia1, ia2, ia3  
ib, ib1, ib2, ib3  
ic, ic1, ic2, ic3  

va, va1, va2, va3  
vb, vb1, vb2, vb3  
vc, vc1, vc2, vc3  

These signals were collected from the Simulink model and used for wavelet-based feature extraction from differnet branch.

## Wavelet Transform Information

The selected wavelet was db4.

The decomposition level used was Level 4.

For each signal, four detail coefficients were extracted:

D1, D2, D3, D4

Where:

- D1 represents the highest-frequency detail component.
- D2 represents the second-level detail component.
- D3 represents the third-level detail component.
- D4 represents the fourth-level detail component.


## Dataset Information

The generated dataset includes both original simulation signals and their corresponding wavelet coefficients.

For each signal, the dataset contains:

Original Signal  
D1 Wavelet Coefficient  
D2 Wavelet Coefficient  
D3 Wavelet Coefficient  
D4 Wavelet Coefficient  

Example:

ia, D1_ia, D2_ia, D3_ia, D4_ia  
ib, D1_ib, D2_ib, D3_ib, D4_ib  
ic, D1_ic, D2_ic, D3_ic, D4_ic  
va, D1_va, D2_va, D3_va, D4_va  
vb, D1_vb, D2_vb, D3_vb, D4_vb  
vc, D1_vc, D2_vc, D3_vc, D4_vc  

The dataset is stored at:

/Santosh/dataset.xlsx
## Phase Selection
Energy: Which phase has more energy
Example: [1-0-0-1] fault : Red phase contains more energy than others.
Ground involment is determined by zero sequence.

Energy formula from paper

Downgrade: 18a




# Wavelet-Based Fault Dataset Generation

## Update by Santosh on 10th May


## Folder Location

The updated Simulink model and generated dataset are stored inside the `Santosh` folder.

    Santosh/Sanepa_overall_model_Fault_18a.slx
    Santosh/Complete_Wavelet_Fault_Dataset.xlsx

## Simulink Settings

| Parameter | Value |
|---|---|
| Sampling Frequency | 1 kHz |
| Sample Time | 0.001 sec |
| Solver | ode45 |
| Stop Time | 12 sec |
| Fault Interval | 1 sec each |

## Fault Timing

| Time Interval | Condition |
|---|---|
| 0–1 sec | NOFAULT |
| 1–2 sec | AG |
| 2–3 sec | BG |
| 3–4 sec | CG |
| 4–5 sec | AB |
| 5–6 sec | BC |
| 6–7 sec | CA |
| 7–8 sec | ABG |
| 8–9 sec | BCG |
| 9–10 sec | CAG |
| 10–11 sec | ABC |
| 11–12 sec | ABCG |

## Signals Stored from Simulink

    ia, ib, ic
    va, vb, vc
    io, vo
    i1, v1
    i2, v2

| Signal | Description |
|---|---|
| `ia`, `ib`, `ic` | Three-phase currents |
| `va`, `vb`, `vc` | Three-phase voltages |
| `io`, `vo` | Zero-sequence current and voltage |
| `i1`, `v1` | Positive-sequence current and voltage |
| `i2`, `v2` | Negative-sequence current and voltage |

## Wavelet Feature Extraction

Discrete Wavelet Transform is applied to each signal using the following settings:

| Parameter | Value |
|---|---|
| Mother Wavelet | db4 |
| Decomposition Level | 4 |
| Window Size | 5 |

Each signal is decomposed into four detail levels:

    D1, D2, D3, D4

Wavelet energy is calculated using:

    E = sum(D(n)^2)

For each signal, four energy features are extracted:

    D1_Energy, D2_Energy, D3_Energy, D4_Energy

## Dataset Format

The final dataset is saved as:

    Santosh/Complete_Wavelet_Fault_Dataset.xlsx

The dataset contains 48 feature columns:

    12 signals × 4 DWT levels = 48 features

Example feature columns:

    ia_D1_Energy, ia_D2_Energy, ia_D3_Energy, ia_D4_Energy
    ib_D1_Energy, ib_D2_Energy, ib_D3_Energy, ib_D4_Energy
    ic_D1_Energy, ic_D2_Energy, ic_D3_Energy, ic_D4_Energy
    io_D1_Energy, io_D2_Energy, io_D3_Energy, io_D4_Energy
    i1_D1_Energy, i1_D2_Energy, i1_D3_Energy, i1_D4_Energy
    i2_D1_Energy, i2_D2_Energy, i2_D3_Energy, i2_D4_Energy
    va_D1_Energy, va_D2_Energy, va_D3_Energy, va_D4_Energy
    vb_D1_Energy, vb_D2_Energy, vb_D3_Energy, vb_D4_Energy
    vc_D1_Energy, vc_D2_Energy, vc_D3_Energy, vc_D4_Energy
    vo_D1_Energy, vo_D2_Energy, vo_D3_Energy, vo_D4_Energy
    v1_D1_Energy, v1_D2_Energy, v1_D3_Energy, v1_D4_Energy
    v2_D1_Energy, v2_D2_Energy, v2_D3_Energy, v2_D4_Energy

The final column is:

    Fault_Label

## Fault Classes

    NOFAULT, AG, BG, CG, AB, BC, CA, ABG, BCG, CAG, ABC, ABCG

## Summary

The Simulink model runs for 12 seconds. The first second is used for no-fault data, and each fault condition is applied for 1 second. Current, voltage, zero-sequence, positive-sequence, and negative-sequence signals are stored from Simulink. DWT energy features are extracted in MATLAB and saved as an Excel dataset for machine learning fault classification.
