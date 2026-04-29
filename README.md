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