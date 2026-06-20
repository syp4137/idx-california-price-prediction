# IDX California Price Prediction

## Project Overview

This repository contains my individual project work for the IDX Exchange Data Science Internship.

The goal of this project is to build a machine learning workflow to predict the final sale price, or `ClosePrice`, of single-family residential properties in California using historical CRMLS sold property data.

This project includes data exploration, preprocessing, feature engineering, model training, evaluation, and documentation of results.

## Project Objective

The main objective is to predict the close price of a residential property based on its characteristics, such as:

* Living area
* Number of bedrooms
* Number of bathrooms
* Lot size
* Property age
* Geographic location
* Other available MLS property features

The target variable is:
```text
ClosePrice
```

## Dataset

The dataset comes from CRMLS sold property records provided through IDX Exchange.
Raw data files are not included in this repository because they are private source data and should not be uploaded to GitHub.

The analysis will focus on:

```text
PropertyType = Residential
PropertySubType = SingleFamilyResidence
```

Only sold or closed records will be used for model training and evaluation because these records contain confirmed `ClosePrice` values.

## Repository Structure

```text
idx-california-price-prediction/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_advanced_models.ipynb
│   └── 06_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── docs/
│   ├── metadata_notes.md
│   ├── weekly_updates.md
│   └── project_notes.md
│
├── outputs/
│   ├── figures/
│   └── metrics/
│
├── models/
│
└── data/
    └── README.md
```

## Methodology

The project workflow includes the following steps:

1. Load and inspect CRMLS sold property data
2. Filter the dataset to residential single-family properties
3. Explore key variables and price distributions
4. Handle missing values and inconsistent data types
5. Encode categorical variables
6. Create additional features such as property age and bed/bath ratio
7. Split the data into training and testing sets
8. Train baseline and advanced machine learning models
9. Evaluate model performance using multiple metrics
10. Document findings and prepare results for presentation

## Models

The following models will be tested and compared:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting models such as XGBoost or LightGBM

Additional models may be added as the project progresses.

## Evaluation Metrics

Model performance will be evaluated using:

* R-squared
* RMSE
* MAPE
* MdAPE

The evaluation will also consider model behavior across different price ranges and geographic areas.

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/idx-california-price-prediction.git
```

2. Move into the project folder:

```bash
cd idx-california-price-prediction
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

4. Install required packages:

```bash
pip install -r requirements.txt
```

5. Place the raw CRMLS files locally in:

```text
data/raw/
```

Raw data should not be committed to GitHub.

6. Run the notebooks in order:

```text
notebooks/01_exploration.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_baseline_model.ipynb
notebooks/04_model_comparison.ipynb
notebooks/05_advanced_models.ipynb
notebooks/06_evaluation.ipynb
```

## Current Progress

Progress will be updated weekly in:

```text
docs/weekly_updates.md
```

## Data Privacy Note

This repository does not include raw CSV files, source data, FTP credentials, or any private IDX Exchange data.
Only code, documentation, model outputs, and non-sensitive summary results will be uploaded.

## Author

Soyeon Park
Data Science Intern, IDX Exchange
M.S. in Statistics and Data Science, Northwestern University
