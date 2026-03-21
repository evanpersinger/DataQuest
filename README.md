# DataQuest

A machine learning project analyzing ER wait times, built for the Western AI Data Quest hackathon.

## Overview

This project explores and models emergency room wait time data using a combination of traditional machine learning and deep learning approaches. The analysis includes exploratory data analysis, feature engineering, and model evaluation.

## Dataset

ER Wait Time Dataset containing healthcare-related metrics for predicting wait times and improving patient outcomes.

## Technologies

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, torch
- **Visualization**: matplotlib, seaborn
- **Notebooks**: Jupyter

## Project Structure

- `notebook.ipynb` - Main analysis and model development
- `ER_Wait_Time_Dataset.csv` - Source dataset
- `pyproject.toml` - Project dependencies and configuration

## Setup

Install dependencies using uv pip:

```bash
uv pip install -r requirements.txt
```

Or install directly from pyproject.toml:

```bash
uv pip install -e .
```

## Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook notebook.ipynb
```

## Requirements

- Python 3.9+
