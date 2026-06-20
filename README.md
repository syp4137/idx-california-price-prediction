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
