# RT-IoT2022 Anomaly Detection: Logistic Regression and Quantized Autoencoders

- [RT-IoT2022 Anomaly Detection: Logistic Regression and Quantized Autoencoders](#rt-iot2022-anomaly-detection-logistic-regression-and-quantized-autoencoders)
- [Overview](#overview)
- [Abstract](#abstract)
- [Data Summary](#data-summary)
- [Methodology](#methodology)
  - [1. Exploratory Data Analysis (01\_eda.ipynb)](#1-exploratory-data-analysis-01_edaipynb)
  - [2. Baseline Supervised Model (02\_baseline\_lr.ipynb)](#2-baseline-supervised-model-02_baseline_lripynb)
  - [3. Autoencoder-Based Anomaly Detection (03\_autoencoders.ipynb)](#3-autoencoder-based-anomaly-detection-03_autoencodersipynb)
- [Implementation](#implementation)
- [Key Findings](#key-findings)
  - [1. The Dataset is Highly Separable](#1-the-dataset-is-highly-separable)
  - [2. Logistic Regression is a Strong Baseline](#2-logistic-regression-is-a-strong-baseline)
  - [3. Autoencoders Perform Competitively](#3-autoencoders-perform-competitively)
  - [4. Quantization Maintains Detection Performance](#4-quantization-maintains-detection-performance)
- [References](#references)
- [Author](#author)

# Overview

This project investigates anomaly detection in Internet of Things (IoT) network traffic using the **RT-IoT2022 dataset**. The goal is to evaluate how effectively different machine learning approaches can distinguish **normal network activity from malicious attacks**.

We compare two primary modeling paradigms:

- **Classical supervised learning:** Logistic Regression
- **Deep learning for anomaly detection:** Autoencoders and Quantized Autoencoders

The project also includes extensive **exploratory data analysis (EDA)** to understand the structure of the dataset, identify informative features, and analyze separability between normal and attack traffic before model training.

The motivation for this work is to evaluate whether **lightweight deep learning approaches such as quantized autoencoders can achieve competitive detection performance while remaining suitable for deployment in resource-constrained IoT environments.**

Full Report: https://github.com/AnhJoe/Real-Time_IoT_2022_Quantized_Auto_Encoder/blob/main/outputs/reports/From-Classical-to-Quantized-Deep-Learning--Anomaly-Detection-in-IoT-Networks-Using-Logistic-Regression-and-Quantized-Autoencoders.pdf

# Abstract

The rapid growth of Internet of Things (IoT) devices has significantly expanded the attack surface of modern networks, creating new challenges for intrusion detection systems. IoT environments often consist of heterogeneous devices with diverse communication patterns and limited computational resources, making it difficult to deploy traditional security mechanisms. This study investigates both supervised and unsupervised machine learning approaches for detecting anomalous network traffic within the **RT-IoT 2022 dataset**, with a focus on developing models that are both accurate and suitable for potential edge deployment.

The analysis begins with **exploratory data analysis (EDA)** to understand the structure and characteristics of the dataset. The results reveal strong class imbalance between normal and attack traffic, substantial multicollinearity among network flow features, and clear separability between many attack and normal traffic patterns. These properties suggest that both classical classification models and anomaly detection techniques may perform effectively in this setting.

We first establish **logistic regression models** as supervised baselines. Three variants were evaluated: a model trained using all features, a LASSO-based feature selection model, and a stepwise-selected feature model. Logistic regression provides an interpretable and computationally lightweight approach while maintaining strong predictive performance, demonstrating that even simple models can effectively detect malicious traffic when the feature space contains informative signals.

Next, we investigate **autoencoder-based anomaly detection**, an unsupervised approach that learns the structure of normal network traffic without requiring labeled attack data. The baseline autoencoder is trained exclusively on normal traffic and uses reconstruction error as an anomaly score. A threshold selected using validation-set F1 maximization converts reconstruction errors into attack predictions. The model achieves strong detection performance, indicating that deviations from learned normal traffic patterns provide a reliable signal for identifying anomalies.

To evaluate deployment-friendly architectures, we further explore **quantized autoencoder variants**, including a half-precision model (QAE-f16) and an 8-bit dynamically quantized model (QAE-u8). These models significantly reduce numerical precision and memory requirements compared with the standard 32-bit baseline. Despite this reduction in precision, both quantized models maintain nearly identical detection performance. The QAE-u8 model introduces a shifted reconstruction error scale requiring a different detection threshold, but classification metrics remain largely unchanged. These findings suggest that quantized autoencoders can retain strong anomaly detection capability while offering substantial efficiency gains.

While the results demonstrate promising performance, several limitations should be considered. The dataset lacks certain documented traffic classes, including Amazon Alexa traffic, which reduces the diversity of benign behaviors available for training. Additionally, strong feature correlations and clear class separation within the dataset may simplify the detection task relative to real-world IoT environments.

Overall, this study shows that both supervised logistic regression and unsupervised autoencoder models can effectively detect anomalous IoT network traffic. Moreover, quantization techniques enable neural models to achieve comparable detection performance while significantly reducing model precision requirements, highlighting their potential for **lightweight intrusion detection in resource-constrained IoT and edge computing environments**.


# Data Summary

Source Link: https://archive.ics.uci.edu/dataset/942/rt-iot2022

RT-IoT2022 is a network traffic dataset collected from a real-time IoT environment containing both normal device activity and multiple simulated cyberattacks. The dataset includes traffic generated by IoT devices alongside attack scenarios such as brute-force SSH attempts, DDoS attacks (e.g., Hping and Slowloris), and network scanning patterns. Network flows are captured bidirectionally using the Zeek monitoring framework with the Flowmeter plugin, producing structured flow-level features that characterize packet behavior, timing, and protocol dynamics.

In this project, RT-IoT2022 serves as the foundation for developing and evaluating intrusion detection approaches—particularly anomaly detection using autoencoders—aimed at identifying malicious behavior within real-time IoT network traffic.

1) **Predictor (Feature) Variables**: The RT-IoT2022 dataset contains **83 predictor features** describing network traffic flows generated by real-time IoT devices. Rather than representing unrelated variables, these features follow a systematic naming convention built from combinations of directional prefixes and statistical or behavioral suffixes. Each feature captures a measurable property of packet flows—such as volume, timing, payload characteristics, protocol behavior, or connection state—computed either for one traffic direction or for the entire bidirectional flow.

    This structured design allows the dataset to encode network behavior at multiple granularities (packet, flow, and temporal statistics), making it well suited for intrusion detection and anomaly detection tasks. Understanding the prefix–suffix pattern is essential for interpreting the predictors.
   
    **Prefixes (traffic scope / direction)**:
    - fwd_ (forward): statistics computed for packets traveling from the origin/source to the responder/destination.
    - bwd_ (backward): statistics computed for packets traveling from the responder/destination back to the origin/source.
    - flow_ (flow-level): aggregate statistics computed across the entire bidirectional communication flow.
    - active_ / idle_: metrics describing active transmission periods versus inactive (silent) intervals within a connection.
    - (No prefix): global or contextual attributes describing protocol, ports, or overall flow characteristics.
    
    **Suffixes (measurement type / statistical summary)**:
    - .min — minimum observed value within the flow.
    - .max — maximum observed value within the flow.
    - .tot — total accumulated value.
    - .avg — arithmetic mean across packets/events.
    - .std — standard deviation measuring variability.
    - _per_sec — rate-based metric normalized by time.
    - _flag_count — count of packets containing specific TCP control flags.
    - _bytes, _pkts, _payload — volume-based measurements describing byte counts, packet counts, or payload sizes.
    - _iat — inter-arrival time statistics between packets.
    - _window_size — TCP window characteristics reflecting transport-layer behavior.

    This prefix–suffix structure enables consistent interpretation across the 83 features while allowing the same behavioral concept (e.g., payload size or timing variability) to be analyzed across traffic directions and statistical summaries.

2) **Outcome (Response) Variable**: Attack_type (categorical)
    - Attack patterns:
        - DOS_SYN_Hping
        - ARP_poisioning
        - NMAP_UDP_SCAN
        - NMAP_XMAS_TREE_SCAN
        - NMAP_OS_DETECTION
        - NMAP_TCP_scan
        - DDOS_Slowloris
        - Metasploit_Brute_Force_SSH
        - NMAP_FIN_SCAN
    - Normal Patterns:
        - MQTT
        - Thing_speak
        - Wipro_bulb_Dataset
        - Amazon-Alexa (**MISSING FROM DATASET**)


# Methodology

The analysis follows a structured pipeline:

## 1. Exploratory Data Analysis (01_eda.ipynb)

Several techniques were used to understand the feature space:

- **Effect Size (Cohen’s d)**  
  Used to identify features that best separate normal and attack traffic.

- **Mutual Information**  
  Quantifies nonlinear dependence between features and the target label.

- **Principal Component Analysis (PCA)**  
  Reduces dimensionality and helps visualize variance structure.

- **Unsupervised Clustering (K-Means)**  
  Explores whether natural groupings exist within the network traffic.

These analyses help determine whether the dataset exhibits **clear separation between classes** and guide feature selection for downstream models.


## 2. Baseline Supervised Model (02_baseline_lr.ipynb)

A **Logistic Regression classifier** is used as a baseline model.

Reasons for selecting logistic regression:

- Strong performance on tabular data
- Interpretable coefficients
- Computationally efficient
- Widely used baseline in intrusion detection research

Feature selection techniques were also evaluated:

- **LASSO Logistic Regression**
- **Stepwise Feature Selection**


## 3. Autoencoder-Based Anomaly Detection (03_autoencoders.ipynb)

To evaluate unsupervised detection methods, we trained:

- **Baseline Autoencoder**
- **Quantized Autoencoder (f16 precision)**
- **Quantized Autoencoder (u8 precision)**

Autoencoders learn **normal traffic patterns** and detect anomalies using reconstruction error.  
Quantization reduces model precision to improve **memory efficiency and deployment feasibility for IoT devices.**


# Implementation

1. Create your virtual environment

2. Install dependencies in requirements.txt

3. Run the ipynb notebooks in notebooks/ in order

4. Raw data is downloaded and saved into data/raw

5. Processed data is saved into data/processed

6. Artifacts/metrics are saved into data/artifacts

7. If changes are made to ipynb notebooks and you wish to render quarto reports,

- Install quarto at https://quarto.org/docs/download/index.html 
- Run: quarto `install tinytex`
- cd to notebooks/ and run: `quarto convert *.ipynb` to convert ipynb to qmd
- Run: `quarto render --to pdf` to render pdf report based on _quarto.yml (this will take some time)
- Rendered reports are saved into outputs/reports 

8. Changes to introduction section can be made in index.qmd

9. Changes to conclusion section can be made in notebooks/04_conclusion.qmd 

# Key Findings

## 1. The Dataset is Highly Separable

Exploratory analysis and model results indicate that **normal and attack traffic are strongly separable in the feature space**.

Most models achieved extremely high detection performance:

- **ROC-AUC ≈ 0.996**
- **PR-AUC ≈ 0.9995**
- **F1 Score > 0.995**

## 2. Logistic Regression is a Strong Baseline

The baseline logistic regression model performed exceptionally well:

- ROC-AUC ≈ **0.996**
- PR-AUC ≈ **0.9999**
- Accuracy ≈ **99.5%**

Feature selection with **LASSO** reduced the model to **23 features** while maintaining nearly identical performance.

This indicates that **a relatively small subset of features captures most of the predictive signal**.

## 3. Autoencoders Perform Competitively

The autoencoder models also achieved strong performance:

- ROC-AUC ≈ **0.996**
- PR-AUC ≈ **0.9995**

Although slightly below the supervised logistic regression models, they remain **highly effective for anomaly detection tasks**.

## 4. Quantization Maintains Detection Performance

Quantized autoencoder models (**QAE-f16 and QAE-u8**) show **minimal performance degradation** relative to the baseline autoencoder.

This suggests that:

- **model precision can be reduced**
- **memory footprint can be lowered**
- **inference efficiency can be improved**

while still maintaining strong anomaly detection capability.

This makes quantized autoencoders particularly attractive for **deployment in resource-constrained IoT environments.**


# References

1. Almadhor, A., Alsubai, S., Kryvinska, N., Al Hejaili, A., Ayari, M., Bouallegue, B., & Abbas, S. (2025). Evaluating large transformer models for anomaly detection of resource-constrained IoT devices for intrusion detection system. Scientific Reports, 15, 37972. https://www.semanticscholar.org/paper/Evaluating-large-transformer-models-for-anomaly-of-Almadhor-Alsubai/ac425cacf7c3d9e729fe7dd8d2242091f24609e4
   
2. Chythanya, K. R., Tuteja, G., Gupta, S., Govindrao, P. S., Mahajan, S., & Singh, A. R. (2025). Efficient anomaly detection in IoT networks using logistic regression and SMOTE. Proceedings of the 6th International Conference for Emerging Technology (INCET). https://www.semanticscholar.org/paper/Efficient-Anomaly-Detection-in-IoT-Networks-Using-Chythanya-Tuteja/f46e4359ac9708b658269baeda3bb7207779adb5 

3. Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). Lawrence Erlbaum Associates. https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00863/full
   
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: Data mining, inference, and prediction (2nd ed.). Springer. https://link.springer.com/book/10.1007/978-0-387-84858-7 
   
5. IBM Security. (2025). Cost of a data breach report 2025: The AI oversight gap. IBM Corporation. https://www.ibm.com/reports/data-breach

6. Patel, N. D., Rao, V. S., & Singh, A. (2024). QDNN-IDS: Quantized deep neural network based computational strategy for intrusion detection in IoT. IEEE Silchar Subsection Conference (SILCON). https://www.semanticscholar.org/paper/QDNN-IDS%3A-Quantized-Deep-Neural-Network-based-for-Patel-Rao/cb6e0039bf8bb4c92b3e48304695362f7fae7e44 

7. Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information: Criteria of max-dependency, max-relevance, and min-redundancy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), 1226–1238. https://ieeexplore.ieee.org/document/1453511
   
8. Putrada, A. G., & Ilhami, D. A. S. (2024). G-mean for optimum threshold in anomaly detection with autoencoder: Cyber security on the RT-IoT2022 dataset. IEEE 22nd Student Conference on Research and Development (SCOReD). https://www.semanticscholar.org/paper/G-Mean-for-Optimum-Threshold-in-Anomaly-Detection-Putrada-Ilhami/c0e1552ff8bacfba87ecbd65bc16bc80405aa68f 

9.  Russell, S. J., & Norvig, P. (2021). Artificial intelligence: A modern approach (4th ed.). Pearson. https://api.pageplace.de/preview/DT0400.9781292401171_A41586057/preview-9781292401171_A41586057.pdf 
   
10. Sharmila, B. S., & Nagapadma, R. (2023). Quantized autoencoder intrusion detection system for anomaly detection in resource-constrained IoT devices using RT-IoT2022 dataset. Cybersecurity, 6(41). https://www.semanticscholar.org/paper/Quantized-autoencoder-(QAE)-intrusion-detection-for-Sharmila-Nagapadma/753f6ede01b4acaa325e302c38f1e0c1ade74f5b

11. Tejaswini, R., Abinaya, P., Anuprabha, S. S., Chidambaram, S. K., Arya, S., Choukiker, Y. K., & Bhowmick, A. (2025). Anomaly detection in IoT networks: A deep learning approach using autoencoders. IEEE Access, 13. https://ieeexplore.ieee.org/document/11077140 

# Author

Joe T. Nguyen, M.S. Data Science, UC Irvine, CA, US
