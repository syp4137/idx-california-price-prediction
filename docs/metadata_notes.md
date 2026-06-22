# Key Feature Notes Based on Available Columns

## 1. Target Variable

| Column       | Meaning                                          | Modeling Use    |
| ------------ | ------------------------------------------------ | --------------- |
| `ClosePrice` | Final sale price paid by the buyer to the seller | Target variable |

`ClosePrice` is the value the model is trying to predict. It should never be included as an input feature.

---

## 2. Filtering Columns

| Column            | Meaning                                                                                    | Modeling Use                            |
| ----------------- | ------------------------------------------------------------------------------------------ | --------------------------------------- |
| `PropertyType`    | Broad property category, such as Residential, Lease, Income, Land, etc.                    | Use for filtering                       |
| `PropertySubType` | More specific property type, such as Single Family Residence, Condominium, Townhouse, etc. | Use for filtering                       |
| `MlsStatus`       | Local MLS listing status                                                                   | Use for filtering/checking sold records |

For this project, the modeling dataset should focus on:

```text
PropertyType = Residential
PropertySubType = SingleFamilyResidence
```

The model should be trained on sold/closed records with valid `ClosePrice`.

---

## 3. Strong Feature Candidates for Modeling

These are the main columns that can be used as input features because they describe the property itself, its size, location, or physical characteristics.

### A. Property Size and Structure

| Column                   | Meaning                                                       | Modeling Use                 |
| ------------------------ | ------------------------------------------------------------- | ---------------------------- |
| `LivingArea`             | Total livable area within the structure                       | Strong numeric feature       |
| `BuildingAreaTotal`      | Total structure area, including finished and unfinished areas | Possible numeric feature     |
| `AboveGradeFinishedArea` | Finished area above ground level                              | Possible numeric feature     |
| `BelowGradeFinishedArea` | Finished area below ground level                              | Possible numeric feature     |
| `BedroomsTotal`          | Total number of bedrooms                                      | Strong numeric feature       |
| `BathroomsTotalInteger`  | Total number of bathrooms as an integer                       | Strong numeric feature       |
| `Stories`                | Number of floors/stories                                      | Possible numeric feature     |
| `Levels`                 | Number/type of levels in the property                         | Possible categorical feature |
| `MainLevelBedrooms`      | Number of bedrooms on the main level                          | Possible numeric feature     |

### B. Lot Size

| Column              | Meaning                            | Modeling Use              |
| ------------------- | ---------------------------------- | ------------------------- |
| `LotSizeAcres`      | Lot size measured in acres         | Strong numeric feature    |
| `LotSizeSquareFeet` | Lot size measured in square feet   | Strong numeric feature    |
| `LotSizeArea`       | Lot size area, depending on unit   | Possible numeric feature  |
| `LotSizeDimensions` | Text description of lot dimensions | Usually not used directly |

Use only one or carefully selected lot-size variable to avoid duplicate information. For example, `LotSizeSquareFeet` and `LotSizeAcres` represent similar information in different units.

### C. Property Age and Construction

| Column              | Meaning                                   | Modeling Use                                       |
| ------------------- | ----------------------------------------- | -------------------------------------------------- |
| `YearBuilt`         | Year the property was built               | Strong numeric feature                             |
| `NewConstructionYN` | Whether the property is newly constructed | Useful boolean feature                             |
| `BuilderName`       | Name of builder or builder tract          | Possible categorical feature, but high-cardinality |

A useful engineered feature is:

```text
PropertyAge = SaleYear - YearBuilt
```

### D. Location

| Column            | Meaning                                    | Modeling Use                                       |
| ----------------- | ------------------------------------------ | -------------------------------------------------- |
| `Latitude`        | Geographic latitude                        | Strong numeric feature                             |
| `Longitude`       | Geographic longitude                       | Strong numeric feature                             |
| `City`            | City of the property                       | Useful categorical feature                         |
| `CountyOrParish`  | County or regional authority               | Useful categorical feature                         |
| `PostalCode`      | ZIP code or postal code                    | Useful categorical feature                         |
| `StateOrProvince` | State abbreviation                         | Usually constant if only California                |
| `MLSAreaMajor`    | MLS-defined market area                    | Useful categorical feature                         |
| `SubdivisionName` | Neighborhood, community, complex, or tract | Possible categorical feature, but high-cardinality |

Location is likely one of the most important groups of features because property prices vary strongly by area.

### E. Amenities and Property Features

| Column             | Meaning                                 | Modeling Use                 |
| ------------------ | --------------------------------------- | ---------------------------- |
| `PoolPrivateYN`    | Whether the property has a private pool | Useful boolean feature       |
| `ViewYN`           | Whether the property has a view         | Useful boolean feature       |
| `WaterfrontYN`     | Whether the property is waterfront      | Useful boolean feature       |
| `BasementYN`       | Whether the property has a basement     | Useful boolean feature       |
| `FireplaceYN`      | Whether the property has a fireplace    | Useful boolean feature       |
| `FireplacesTotal`  | Total number of fireplaces              | Possible numeric feature     |
| `AttachedGarageYN` | Whether the garage is attached          | Useful boolean feature       |
| `GarageSpaces`     | Number of garage spaces                 | Useful numeric feature       |
| `ParkingTotal`     | Total parking spaces                    | Useful numeric feature       |
| `CoveredSpaces`    | Total garage and carport spaces         | Possible numeric feature     |
| `Flooring`         | Type of flooring                        | Possible categorical feature |

Some amenities may have missing values. Boolean features should be checked carefully because missing values may mean unknown rather than false.

### F. School and District Information

| Column                         | Meaning                                  | Modeling Use                 |
| ------------------------------ | ---------------------------------------- | ---------------------------- |
| `ElementarySchool`             | Associated elementary school             | Possible categorical feature |
| `ElementarySchoolDistrict`     | Associated elementary school district    | Useful categorical feature   |
| `MiddleOrJuniorSchool`         | Associated middle/junior school          | Possible categorical feature |
| `MiddleOrJuniorSchoolDistrict` | Associated middle/junior school district | Useful categorical feature   |
| `HighSchool`                   | Associated high school                   | Possible categorical feature |
| `HighSchoolDistrict`           | Associated high school district          | Useful categorical feature   |

School district features may be useful because school boundaries can affect home prices. However, individual school names may be high-cardinality and should be encoded carefully.

### G. HOA and Fees

| Column                    | Meaning                      | Modeling Use                 |
| ------------------------- | ---------------------------- | ---------------------------- |
| `AssociationFee`          | HOA fee amount               | Possible numeric feature     |
| `AssociationFeeFrequency` | Frequency of HOA fee payment | Possible categorical feature |

HOA information can be useful, but missing values should be handled carefully.

---

## 4. Use With Caution

These columns may contain useful information, but they should be used carefully because they may introduce leakage, depend on prediction timing, or require careful preprocessing.

| Column                     | Reason for Caution                                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `DaysOnMarket`             | May be useful if known at prediction time, but can cause leakage if final DOM is only known after sale                                           |
| `ListingContractDate`      | Useful for market timing or seasonality, but not a physical property characteristic                                                              |
| `PurchaseContractDate`     | Usually known after an offer is accepted, so may not be available at prediction time                                                             |
| `ContractStatusChangeDate` | May reflect transaction progress, so it can introduce leakage depending on prediction timing                                                     |
| `TaxAnnualAmount`          | May be highly correlated with property value; useful but should be reviewed carefully                                                            |
| `TaxYear`                  | Useful only together with tax-related features                                                                                                   |
| `UnparsedAddress`          | Full address is high-cardinality and should not be used directly; location should be captured through ZIP, city, county, latitude, and longitude |
| `StreetNumberNumeric`      | Usually not meaningful as a predictive feature by itself                                                                                         |

---

## 5. Columns to Exclude From Machine Learning Features

These columns should not be used as model input features.

### A. Target or Leakage Columns

| Column              | Reason for Exclusion                                                                      |
| ------------------- | ----------------------------------------------------------------------------------------- |
| `ClosePrice`        | Target variable                                                                           |
| `ListPrice`         | Leakage risk; can artificially inflate model performance                                  |
| `OriginalListPrice` | Leakage risk; can artificially inflate model performance                                  |
| `CloseDate`         | Known after the transaction closes; should be used for time split, not as a model feature |

`ListPrice` and `OriginalListPrice` may still be useful for exploratory analysis, but they should not be included in the training feature set.

### B. Identifier Columns

| Column              | Reason for Exclusion                         |
| ------------------- | -------------------------------------------- |
| `ListingKey`        | Unique identifier, not a predictive feature  |
| `ListingKeyNumeric` | Numeric version of listing identifier        |
| `ListingId`         | Listing identifier, not a predictive feature |

Identifiers should be used for tracking, joining, or deduplication, not for prediction.

### C. Agent and Office Columns

| Column                  | Reason for Exclusion                                                |
| ----------------------- | ------------------------------------------------------------------- |
| `ListAgentEmail`        | Personal/contact information; not appropriate for modeling          |
| `ListAgentFirstName`    | High-cardinality agent field                                        |
| `ListAgentLastName`     | High-cardinality agent field                                        |
| `ListAgentFullName`     | High-cardinality agent field                                        |
| `CoListAgentFirstName`  | High-cardinality agent field                                        |
| `CoListAgentLastName`   | High-cardinality agent field                                        |
| `BuyerAgentFirstName`   | Buyer-side information may not be known at prediction time          |
| `BuyerAgentLastName`    | Buyer-side information may not be known at prediction time          |
| `BuyerAgentMlsId`       | Agent identifier                                                    |
| `CoBuyerAgentFirstName` | Buyer-side/co-buyer information may not be known at prediction time |
| `ListOfficeName`        | Brokerage field; high-cardinality                                   |
| `BuyerOfficeName`       | Buyer-side office may not be known at prediction time               |
| `CoListOfficeName`      | Brokerage field; high-cardinality                                   |
| `BuyerAgentAOR`         | Agent association field, not a core property characteristic         |
| `ListAgentAOR`          | Agent association field, not a core property characteristic         |
| `BuyerOfficeAOR`        | Office association field, not a core property characteristic        |

Agent and office columns may be useful for separate brokerage-level analysis, but they are not recommended as direct features for this property price prediction model.

### D. Irrelevant or Less Relevant Columns for Residential Price Prediction

| Column         | Reason for Exclusion                                                                    |
| -------------- | --------------------------------------------------------------------------------------- |
| `BusinessType` | More relevant to business/commercial properties, not single-family residential modeling |

---

## 6. Recommended Initial Feature Set

For the first clean model, I would start with the following features:

### Numeric Features

```text
LivingArea
BedroomsTotal
BathroomsTotalInteger
LotSizeSquareFeet
YearBuilt
Latitude
Longitude
GarageSpaces
ParkingTotal
FireplacesTotal
AssociationFee
TaxAnnualAmount
```

### Boolean Features

```text
ViewYN
WaterfrontYN
BasementYN
PoolPrivateYN
AttachedGarageYN
FireplaceYN
NewConstructionYN
```

### Categorical Features

```text
City
CountyOrParish
PostalCode
MLSAreaMajor
PropertySubType
ElementarySchoolDistrict
MiddleOrJuniorSchoolDistrict
HighSchoolDistrict
Levels
Flooring
AssociationFeeFrequency
```

### Engineered Features

```text
PropertyAge = SaleYear - YearBuilt
HasGarage = GarageSpaces > 0
HasFireplace = FireplaceYN or FireplacesTotal > 0
LogLivingArea = log(LivingArea)
LogLotSize = log(LotSizeSquareFeet)
```

---

## 7. Final Modeling Notes

The model should focus on property characteristics and location-based features. Price-related fields such as `ListPrice` and `OriginalListPrice` should be excluded because they introduce leakage. Transaction outcome fields and identifiers should also be excluded from the feature set.

A safe first modeling dataset should include physical property features, location features, selected amenity features, and carefully encoded categorical variables.
