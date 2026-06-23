# Household Power Forecasting

Multi-variate time series forecasting for household electricity consumption using LSTM, Transformer, and PatchConvFormer.

## Datasets

This project requires two datasets. Please download them and place them in the `data/` directory.

1.  **Electricity Dataset**: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
    -   Download the file `household_power_consumption.txt` and save it to `data/`.

2.  **Weather Dataset**: [Data.gouv.fr - Données climatologiques de base mensuelles](https://www.data.gov.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles)
    -   Download the file `MENSQ_92_previous-1950-2024.csv` and save it to `data/`.

> **Note**: The raw data is at minute-level. The `data_preprocessing.py` script will aggregate it to daily level and merge it with weather data.

## Models

| Model | Description |
|-------|-------------|
| LSTM | Baseline RNN with gated recurrent units |
| Transformer | Self-attention based encoder |
| PatchConvFormer | Proposed model: Conv1D for local patterns + patch-based Transformer (inspired by PatchTST) |

## Requirements

- Python, PyTorch, NumPy, pandas, scikit-learn, matplotlib

## Usage

```bash
# 1. Download datasets and place them in data/ folder

# 2. Preprocess data (minute-level → daily-level)
python data_preprocessing.py

# 3. Run 5 random seeds
bash run_5seeds.sh
