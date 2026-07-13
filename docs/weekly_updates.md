## Week 2 — Data Exploration

### Completed

* Loaded 6 months of CRMLS sold property data from local CSV files:

  * Dec 2025
  * Jan 2026
  * Feb 2026
  * Mar 2026
  * Apr 2026
  * May 2026

* Combined dataset size:

  * Raw dataset: 124,404 rows and 79 columns
  * Filtered dataset: 61,727 rows after keeping only `Residential` and `SingleFamilyResidence` records

* Confirmed that all records in the loaded dataset have `MlsStatus = Closed`.

* Reviewed the target variable, `ClosePrice`:

  * No missing values
  * No non-positive values
  * Highly right-skewed distribution
  * Some extreme values require additional review during preprocessing

* Checked missing values across all columns:

  * Several columns are 100% missing and will be excluded from the baseline model
  * Several additional columns have very high missing rates and will likely be excluded from the first model

* Checked duplicate listing records:

  * Found 62 duplicated `ListingKey` values, creating 70 extra rows
  * Most duplicates appear across different monthly files
  * A deduplication rule is needed in the preprocessing step

* Checked geographic coordinates:

  * Most records fall within the expected California latitude/longitude range
  * 24 records were flagged as outside the expected California coordinate range

* Explored relationships between key features and `ClosePrice`:

  * `LivingArea`, `BathroomsTotalInteger`, and `BedroomsTotal` showed the strongest monotonic relationships with `ClosePrice`
  * Location-based features such as county, city, ZIP code, and MLS area appear important
  * Price levels vary significantly across counties and cities

* Created additional EDA views:

  * Price per square foot distribution
  * County-level and city-level median prices
  * Monthly sales counts and monthly median prices
  * Extreme value checks for key numeric columns

### Notes

* `ListPrice` and `OriginalListPrice` will not be used in the baseline model because they may introduce target leakage.
* `PricePerSqFt` is useful for EDA, but it should not be used as a model feature because it is derived from `ClosePrice`.
* Actual data cleaning decisions will be implemented in the preprocessing notebook rather than directly in the EDA notebook.

### Next Steps

* Create `02_preprocessing.ipynb`
* Define deduplication rules for repeated `ListingKey` values
* Exclude leakage columns and identifier columns
* Handle missing values and high-missing columns
* Review and handle extreme values
* Create a time-based train/test split using the latest month as the test period
* Prepare the first baseline model


## Week 3 — Data Preprocessing

This week, I continued from `01_exploration.ipynb` and worked on `02_preprocessing.ipynb`.

### Completed

- Created a cleaned base dataset for modeling.
- Kept the project filter:
  - `PropertyType = Residential`
  - `PropertySubType = SingleFamilyResidence`
- Removed rows with missing or invalid `ClosePrice` and `CloseDate`.
- Handled duplicated `ListingKey` values by treating them as repeated or updated copies of the same listing record.
- Cleaned numeric, boolean, and location-related fields.
- Added missing/invalid flags where useful.
- Created a time-based train/test split plan:
  - most recent close month = test set
  - previous X months = training set
  - X is treated as a tunable training window length
- Saved cleaned split files for each candidate X-month training window.
- Added a preprocessing pipeline template for future modeling.

### Key Decision

To avoid data leakage, I did not fit imputation, scaling, rare-category grouping, or one-hot encoding on the full dataset.

Instead, the cleaned CSV is saved before those steps. In the modeling notebook, preprocessing will be fit on the training set only and then applied to the test set.

### Outputs

- `02_preprocessing.ipynb`
- cleaned base CSV
- split plan CSV
- cleaned train/test split files

### Next Steps

- Start the baseline modeling notebook.
- Train a Linear Regression model.
- Compare test R² across different training window lengths.
- Select the best X-month training window based on model performance.


## Week 4 — Baseline Model

- Built a Linear Regression baseline model for predicting `ClosePrice`.
- Used the agreed recent six-month block: five months for training and the following month for evaluation.
- Added older data only for a rolling-origin stability check.
- Used 40 raw features, including property characteristics, location variables, boolean fields, and missing-value flags.
- Excluded leakage-related fields such as `ListPrice`, `OriginalListPrice`, `DaysOnMarket`, and post-close information.
- Built a train-only preprocessing pipeline:
  - median imputation and standardization for numeric features
  - missing-category imputation and one-hot encoding for categorical features
  - most-frequent imputation for boolean features
  - zero imputation for missing-value flags
- Calculated the 0.5th and 99.5th percentile `ClosePrice` bounds from each training set only and applied the same bounds to its evaluation set.
- Ran the same pipeline across five historical cutoffs. The rolling-origin results were stable, with a mean R² of 0.8202 and a standard deviation of 0.0062.
- Trained the final model on December 2025 through April 2026 and evaluated it on May 2026.
- Final test results:
  - R²: 0.8323
  - MAE: about $247K
  - RMSE: about $414K
  - MAPE: 23.04%
  - MdAPE: 16.05%
- Saved the final preprocessing pipeline, model, predictions, and evaluation results.
- My R² was noticeably higher than the other team results, so I plan to compare our feature sets, outlier handling, and test samples to understand the difference.