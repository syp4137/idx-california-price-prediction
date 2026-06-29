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
