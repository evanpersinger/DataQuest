# DataQuest

A machine learning project analyzing ER wait times with an interactive simulator, built for the Western AI Data Quest hackathon.

## Overview

This project predicts emergency room wait times using machine learning models trained on real-world healthcare data. It includes exploratory analysis, feature engineering, model training, and an interactive Streamlit application for simulating wait times under different conditions.

## Dataset

ER Wait Time Dataset containing 5000+ records with patient and hospital metrics for predicting total wait times.

## Technologies

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (Linear Regression, Random Forest, Gradient Boosting)
- **Frontend**: Streamlit
- **Visualization**: matplotlib, seaborn
- **Analysis**: Jupyter

## Project Structure

- `app.py` - Interactive Streamlit application for wait time simulation
- `notebook.ipynb` - Exploratory analysis, model training, and evaluation
- `ER_Wait_Time_Dataset.csv` - Source dataset
- `pyproject.toml` - Project dependencies and configuration

## Setup

Install dependencies using uv:

```bash
uv sync
```

## Usage

For analysis and model development, open the Jupyter notebook:

```bash
jupyter notebook notebook.ipynb
```

## Requirements

- Python 3.9+
