# IDX California Price Prediction

This repository contains an IDX Exchange Data Science Internship project for predicting California single-family residential sale prices from historical CRMLS sold listing data.

The modeling target is `ClosePrice`, the final closed sale price.

## Project Objective

The goal is to build an end-to-end machine learning workflow that can estimate the sale price of a residential property from MLS property characteristics, location fields, and engineered real-estate features.

Core prediction inputs include:

- Living area
- Bedrooms and bathrooms
- Lot size
- Property age and structural attributes
- County, city, postal code, latitude, and longitude
- School district information
- MLS-provided property characteristics and quality flags

## Dataset Source

The source data is monthly CRMLS sold property data provided through IDX Exchange. The local raw files follow this pattern:

```text
data/raw/CRMLSSoldYYYYMM.csv
data/raw/CRMLSSoldYYYYMM_filled.csv
```

The current project data covers monthly sold records from January 2024 through June 2026. Raw CRMLS files are private IDX Exchange source data and should not be uploaded to a public repository.

The project scope is restricted to:

```text
PropertyType = Residential
PropertySubType = SingleFamilyResidence
MlsStatus = Closed
```

An additional public geospatial file is used for school district feature engineering:

```text
data/DistrictAreas2526.geojson
```

## Preprocessing

The preprocessing workflow is implemented mainly in:

```text
notebooks/02_preprocessing.ipynb
notebooks/03-1_x_window_preprocessing_202606.ipynb
```

Main preprocessing steps:

- Load and combine monthly CRMLS sold-property files.
- Filter to residential single-family closed transactions.
- Remove rows with missing or invalid `ClosePrice` and `CloseDate`.
- Deduplicate repeated `ListingKey` records.
- Clean numeric, boolean, date, ZIP, lot size, and location fields.
- Create missing-value and invalid-value flags where useful.
- Exclude identifier columns and leakage-prone fields such as `ListPrice`, `OriginalListPrice`, `DaysOnMarket`, price-per-square-foot fields derived from the target, and post-close information.
- Create time-based train/validation/test splits so that future months are not used to train models for earlier months.
- Fit imputation, scaling, rare-category handling, and one-hot encoding on the training data only.
- Apply outlier filters using training-set-derived `ClosePrice` and price-per-square-foot bounds.

Feature engineering added:

- `PropertyAgeYears`
- `BedBathRatio`
- `LivingAreaPerBedroom`
- `LogLotToLivingAreaRatio`
- `HasGarage`
- `UnifiedSchoolDistrict` from a spatial join between property coordinates and California school district boundaries

## Models Tested

The project tested the following model families:

- Linear Regression baseline
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- Streamlit demo model using HistGradientBoosting with a simplified four-feature input

Primary notebooks:

```text
notebooks/03_baseline_model.ipynb
notebooks/04_model_comparison.ipynb
notebooks/04-1_location_ablation_experiment.ipynb
notebooks/05_advanced_models.ipynb
notebooks/06-1_additional_experiment.ipynb
notebooks/06_evaluation.ipynb
notebooks/09_streamlit_demo_model.ipynb
```

## Best Results

The best full project model is the tuned XGBoost model with log-transformed target and the `Base + Structural Derived + UnifiedSchoolDistrict` feature set.

Final test setup:

- Training window: June 2025 through May 2026
- Test month: June 2026
- Test rows after filtering: 12,538
- Raw feature count: 54

Final test results:

| Model | Feature Set | Target | Test R2 | MAE | RMSE | MdAPE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| XGBoost tuned | Base + Structural Derived + UnifiedSchoolDistrict | log1p | 0.909 | $148,075 | $289,276 | 7.65% |
| Random Forest | Base + UnifiedSchoolDistrict | raw | 0.890 | $157,781 | $317,251 | 7.60% |
| Linear Regression | Base + Structural Derived + UnifiedSchoolDistrict | raw | 0.855 | $215,528 | $364,204 | 14.34% |
| Decision Tree | Base + UnifiedSchoolDistrict | raw | 0.764 | $226,461 | $464,916 | 10.84% |

The final XGBoost model is saved at:

```text
models/week7_advanced_models/xgboost_tuned_pipeline.joblib
```

The simplified Streamlit demo model is intentionally less accurate because it only uses `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, and `LotSizeSquareFeet`.

Streamlit demo model results:

| Model | R2 | MAE | RMSE | MdAPE |
| --- | ---: | ---: | ---: | ---: |
| HistGradientBoosting demo model | 0.423 | $446,435 | $730,459 | 30.03% |

## Repository Structure

```text
idx-california-price-prediction/
|-- app.py
|-- data/
|   |-- README.md
|   |-- raw/
|   |-- processed/
|   `-- DistrictAreas2526.geojson
|-- docs/
|   |-- metadata_notes.md
|   `-- weekly_updates.md
|-- models/
|-- notebooks/
|-- outputs/
|-- requirements.txt
`-- README.md
```

## Setup

From the project directory:

```bash
cd idx-california-price-prediction
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If Jupyter does not automatically detect the virtual environment, register it as a kernel:

```bash
python3 -m ipykernel install --user --name idx-price-prediction --display-name "IDX Price Prediction"
```

## Re-run the Code

To reproduce the workflow, run the notebooks in this order:

1. `notebooks/01_exploration.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03-1_x_window_preprocessing_202606.ipynb`
4. `notebooks/03-2_x_window_baseline_experiment_202606.ipynb`
5. `notebooks/03_baseline_model.ipynb`
6. `notebooks/04_model_comparison.ipynb`
7. `notebooks/04-1_location_ablation_experiment.ipynb`
8. `notebooks/05_advanced_models.ipynb`
9. `notebooks/06-1_additional_experiment.ipynb`
10. `notebooks/06_evaluation.ipynb`
11. `notebooks/09_streamlit_demo_model.ipynb`

Start Jupyter with:

```bash
jupyter notebook
```

The notebooks write model artifacts and summary files to:

```text
models/
outputs/
```

Important output files:

```text
outputs/week7_advanced_models/xgboost_final_test_results.csv
outputs/week7_advanced_models/xgboost_model_comparison.csv
outputs/week8_evaluation/metrics_summary.csv
outputs/week9_streamlit/streamlit_demo_model_metrics.csv
```

## Launch the App

The Streamlit app loads this saved model artifact:

```text
models/week9_streamlit/streamlit_demo_price_pipeline.joblib
```

To launch the app:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

If the model artifact is missing, run `notebooks/09_streamlit_demo_model.ipynb` first.

## Data Privacy

This repository should not include raw private CRMLS source data, FTP credentials, or other private IDX Exchange data in public commits. Keep raw CSV files local under `data/raw/` and commit only code, documentation, non-sensitive summaries, and approved artifacts.

## Author

Soyeon Park  
Data Science Intern, IDX Exchange  
M.S. in Statistics and Data Science, Northwestern University
