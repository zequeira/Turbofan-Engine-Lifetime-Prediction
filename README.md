# Lifetime Prediction of Turbofan Engines

The goal of this project is to develop deep learning models based on LSTM and Bi-LSTM in PyTorch to predict the remaining useful life of turbofan engines.
Two datasets will be employed, i.e., the extensively used CMAPSS [[1]](#1) dataset, and the N-CMAPSS dataset [[2]](#2) that was recently developed to account for more realistic run-to-failure trajectories for a fleet of aircraft engines under real flight conditions.

The developed prognostic predictions models are relevant for the implementation of intelligent maintenance strategies. 
These models would reduce machine downtime, costs, and the risk of potentially catastrophic consequences if systems are not maintained on time.

The **main contribution** of this project is the training and evaluation done with the N-CMAPSS dataset.
This dataset was just released in 2021. And to my knowledge, it has not yet been used to develop models for predicting remaining useful life.

Please check the following [**report**](Documentation/Report.pdf) for a detailed explanation of both datasets, model architectures, and the results.

### Data Preparation

For the CMAPSS dataset, run `python data_preparation_CMAPSS.py` to computer the target variable (i.e, RUL), scale input and target features, and split the test files into validation and test set.
A `train_*.csv`, `val_*.csv`, and `test_*.csv` csv file is created for FD001, FD002, FD003, and FD004.

For the N-CMAPSS dataset run `python data_preparation_N-CMAPSS.py` to load the data from the `DS03.h5` file, scale the input and target features, and split into training, validation, and test dataset.
A `train_DS03.csv`, `val_DS03.csv`, and `test_DS03.csv` csv file is created.

### Model

| Layer Type | No. of Layers | No. Nodes | 
| ---------- | ------------- | --------- |
| Bi-LSTM    | 3             | 100       |
| Dropout (0.5) | 2          | --        |
| FC         | 1             | 200       |

### Results

**CMAPSS Dataset**

| Sub-dataset | Validation (RMSE) | Test (RMSE) |
| ----------- | ----------------- | ----------- |
| FD001       | 32.87             | 37.7        |
| FD002       | 42.1              | 39.6        |
| FD003       | 49.3              | 55.34       |
| FD004       | 71.5              | 59.5        |

**N-CMAPSS Dataset**

| Sub-dataset | Validation (RMSE) | Test (RMSE) |
| ----------- | ----------------- | ----------- |
| DS03        |                   | 8.67        |


### References

<a id="1">[1]</a>
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. 2008 International Conference on Prognostics and Health Management, 1–9. [DOI: https://doi.org/10.1109/PHM.2008.4711414]

<a id="2">[2]</a>
Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2021). Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics. Data, 6(1), 5.