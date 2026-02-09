# Real-Time_IoT_2022_Quantized_Auto_Encoder
This is a replication of a study using Quantized autoencoder (QAE) intrusion detection system for anomaly detection in resource-constrained IoT devices using RT-IoT2022 dataset. See references below.

References:
1) Sharmila, B.S., & Nagapadma, R. (2023). Quantized autoencoder (QAE) intrusion detection system for anomaly detection in resource-constrained IoT devices using RT-IoT2022 dataset. Cybersecurity, 6, 1-15.
  - Source Link: https://www.semanticscholar.org/paper/Quantized-autoencoder-(QAE)-intrusion-detection-for-Sharmila-Nagapadma/753f6ede01b4acaa325e302c38f1e0c1ade74f5b

3/20 Friday deadline for 210p final project
Rubric: 
a) Must be goal-oriented
b) Fully understand the data structure (collection, missing, type, error, etc.), but doesn't necessarily require every feature
c) Data structure accuracy (outliers, interpretability, etc.)
d) Extensive EDA with visuals & summaries that are clean & descriptive that aligns with the goals (heavily weighted for scoring)
e) Modeling with depth and variation (2-3 meaningful models within data context)
f) Interpretation/Discussion/Data insights to show depth
g) Ideally, bring together both inference and predictive modeling (not necessarily feasible for all datasets)


rt-iot2022-qae/
├── README.md
├── requirements.txt
├── .gitignore
├── _quarto.yml
├── report.ipynb                
├── data/
│   ├── RT_IOT2022.csv
├── src/
│   ├── __init__.py
│   ├── config.py                 # constants, paths, random seeds
│   ├── data.py                   # load/split/scale utilities
│   ├── models.py                 # AE + quantization hooks
│   ├── metrics.py                # RE, thresholding, eval metrics
│   └── viz.py                    # plots for RE curves, confusion, etc.
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── reports/                  # Quarto render outputs (HTML/PDF)

