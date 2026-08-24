from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


RANDOM_STATE = 42
TARGET = "ClosePrice"
HIGH_CARDINALITY_CATEGORICAL_COLS = ["City", "PostalCode", "MLSAreaMajor"]
HIGH_CARDINALITY_MIN_FREQUENCY = 20

ZIP_LOCAL_AGGREGATE_NUMERIC_COLS = [
    "zip_median_ppsf_6m",
    "zip_90pct_price_12m",
    "zip_sales_count_6m",
    "local_sales_count_6m",
    "LivingArea_x_zip_median_ppsf_6m",
    "LivingArea_x_local_median_ppsf",
]
DISTRICT_LOCAL_AGGREGATE_NUMERIC_COLS = [
    "district_median_ppsf_12m",
    "district_90pct_price_12m",
    "district_sales_count_12m",
    "LivingArea_x_district_median_ppsf_12m",
]


def make_regular_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_high_cardinality_encoder(
    min_frequency: int = HIGH_CARDINALITY_MIN_FREQUENCY,
) -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse=True,
        )


def split_categorical_features(categorical_features: list[str]) -> tuple[list[str], list[str]]:
    high_cardinality_features = [
        col for col in categorical_features if col in HIGH_CARDINALITY_CATEGORICAL_COLS
    ]
    regular_categorical_features = [
        col for col in categorical_features if col not in HIGH_CARDINALITY_CATEGORICAL_COLS
    ]
    return high_cardinality_features, regular_categorical_features


def local_aggregate_numeric_features(categorical_features: list[str]) -> list[str]:
    features = list(ZIP_LOCAL_AGGREGATE_NUMERIC_COLS)
    if "UnifiedSchoolDistrict" in categorical_features:
        features += DISTRICT_LOCAL_AGGREGATE_NUMERIC_COLS
    return features


def transform_target_for_mode(y: pd.Series, target_transform: str) -> pd.Series | np.ndarray:
    if target_transform == "raw":
        return y
    if target_transform == "log1p":
        return np.log1p(y)
    raise ValueError(f"Unsupported target transform: {target_transform}")


def inverse_predictions_for_mode(
    predictions: np.ndarray,
    target_transform: str,
) -> np.ndarray:
    if target_transform == "raw":
        return np.clip(predictions, 0, None)
    if target_transform == "log1p":
        return np.clip(np.expm1(predictions), 0, None)
    raise ValueError(f"Unsupported target transform: {target_transform}")


class LocalAggregateFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create train-fold-only local price features from ZIP and district history."""

    def __init__(
        self,
        target_col: str = TARGET,
        date_col: str = "CloseDate",
        zip_col: str = "PostalCode",
        district_col: str = "UnifiedSchoolDistrict",
        living_area_col: str = "LivingArea",
        min_count: int = 20,
    ):
        self.target_col = target_col
        self.date_col = date_col
        self.zip_col = zip_col
        self.district_col = district_col
        self.living_area_col = living_area_col
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        history = X.copy()
        if self.target_col in history.columns:
            y_series = pd.to_numeric(history[self.target_col], errors="coerce")
        elif y is not None:
            y_series = pd.Series(y, index=history.index).astype(float)
        else:
            raise ValueError("Local aggregate features require raw ClosePrice in X during fit.")

        history[self.target_col] = y_series
        history[self.date_col] = pd.to_datetime(history.get(self.date_col), errors="coerce")
        history[self.living_area_col] = pd.to_numeric(
            history.get(self.living_area_col),
            errors="coerce",
        )
        history["_ppsf"] = np.where(
            history[self.target_col].gt(0) & history[self.living_area_col].gt(0),
            history[self.target_col] / history[self.living_area_col],
            np.nan,
        )

        max_date = history[self.date_col].max()
        if pd.isna(max_date):
            history_6m = history.copy()
            history_12m = history.copy()
        else:
            history_6m = history[history[self.date_col].ge(max_date - pd.DateOffset(months=6))].copy()
            history_12m = history[history[self.date_col].ge(max_date - pd.DateOffset(months=12))].copy()

        self.global_median_ppsf_ = history["_ppsf"].median()
        self.zip_median_ppsf_6m_ = self._build_group_map(history_6m, self.zip_col, "_ppsf", "median")
        self.zip_90pct_price_12m_ = self._build_group_map(history_12m, self.zip_col, self.target_col, "q90")
        self.zip_sales_count_6m_ = self._build_group_map(history_6m, self.zip_col, self.target_col, "count")
        self.district_median_ppsf_12m_ = self._build_group_map(history_12m, self.district_col, "_ppsf", "median")
        self.district_90pct_price_12m_ = self._build_group_map(history_12m, self.district_col, self.target_col, "q90")
        self.district_sales_count_12m_ = self._build_group_map(history_12m, self.district_col, self.target_col, "count")
        self.global_sales_count_6m_ = len(history_6m)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        zip_values = (
            X_out[self.zip_col].astype("string")
            if self.zip_col in X_out.columns
            else pd.Series(pd.NA, index=X_out.index)
        )
        district_values = (
            X_out[self.district_col].astype("string")
            if self.district_col in X_out.columns
            else pd.Series(pd.NA, index=X_out.index)
        )
        living_area = pd.to_numeric(X_out.get(self.living_area_col), errors="coerce")

        zip_median_ppsf_6m = zip_values.map(self.zip_median_ppsf_6m_).astype(float)
        zip_90pct_price_12m = zip_values.map(self.zip_90pct_price_12m_).astype(float)
        zip_sales_count_6m = zip_values.map(self.zip_sales_count_6m_).astype(float)
        district_median_ppsf_12m = district_values.map(self.district_median_ppsf_12m_).astype(float)
        district_90pct_price_12m = district_values.map(self.district_90pct_price_12m_).astype(float)
        district_sales_count_12m = district_values.map(self.district_sales_count_12m_).astype(float)

        local_median_ppsf = zip_median_ppsf_6m.fillna(district_median_ppsf_12m).fillna(
            self.global_median_ppsf_
        )
        local_sales_count_6m = zip_sales_count_6m.fillna(self.global_sales_count_6m_)

        X_out["zip_median_ppsf_6m"] = zip_median_ppsf_6m
        X_out["zip_90pct_price_12m"] = zip_90pct_price_12m
        X_out["zip_sales_count_6m"] = zip_sales_count_6m
        X_out["district_median_ppsf_12m"] = district_median_ppsf_12m
        X_out["district_90pct_price_12m"] = district_90pct_price_12m
        X_out["district_sales_count_12m"] = district_sales_count_12m
        X_out["local_sales_count_6m"] = local_sales_count_6m
        X_out["LivingArea_x_zip_median_ppsf_6m"] = living_area * zip_median_ppsf_6m
        X_out["LivingArea_x_district_median_ppsf_12m"] = living_area * district_median_ppsf_12m
        X_out["LivingArea_x_local_median_ppsf"] = living_area * local_median_ppsf
        return X_out

    def _build_group_map(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        agg_name: str,
    ) -> pd.Series:
        if group_col not in df.columns:
            return pd.Series(dtype=float)
        grouped = df.dropna(subset=[group_col, value_col]).copy()
        grouped[group_col] = grouped[group_col].astype("string")
        group_sizes = grouped.groupby(group_col, observed=True)[value_col].size()
        valid_groups = group_sizes[group_sizes >= self.min_count].index
        grouped = grouped[grouped[group_col].isin(valid_groups)]
        if grouped.empty:
            return pd.Series(dtype=float)
        if agg_name == "median":
            return grouped.groupby(group_col, observed=True)[value_col].median()
        if agg_name == "q90":
            return grouped.groupby(group_col, observed=True)[value_col].quantile(0.90)
        if agg_name == "count":
            return grouped.groupby(group_col, observed=True)[value_col].size().astype(float)
        raise ValueError(f"Unsupported aggregate: {agg_name}")


def build_local_aggregate_xgboost_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    boolean_features: list[str],
    flag_features: list[str],
    model_params: dict[str, Any],
) -> Pipeline:
    numeric_features = list(
        dict.fromkeys(numeric_features + local_aggregate_numeric_features(categorical_features))
    )
    high_cardinality_features, regular_categorical_features = split_categorical_features(
        categorical_features
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "high_cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", make_high_cardinality_encoder()),
                    ]
                ),
                high_cardinality_features,
            ),
            (
                "regular_cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", make_regular_onehot_encoder()),
                    ]
                ),
                regular_categorical_features,
            ),
            ("bool", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))]), boolean_features),
            ("flag", Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]), flag_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **model_params,
    )
    return Pipeline(
        steps=[
            ("local_aggregates", LocalAggregateFeatureTransformer(target_col=TARGET)),
            ("preprocess", column_transformer),
            ("model", model),
        ]
    )


@dataclass
class FinalBlendPriceModel:
    log1p_model: Pipeline
    raw_model: Pipeline
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    boolean_cols: list[str]
    flag_cols: list[str]
    log1p_weight: float = 0.6
    raw_weight: float = 0.4
    target: str = TARGET
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_prepared = self._prepare_features(X)
        log1p_predictions = inverse_predictions_for_mode(
            self.log1p_model.predict(X_prepared),
            "log1p",
        )
        raw_predictions = inverse_predictions_for_mode(
            self.raw_model.predict(X_prepared),
            "raw",
        )
        return np.clip(
            self.log1p_weight * log1p_predictions + self.raw_weight * raw_predictions,
            0,
            None,
        )

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_prepared = X.copy()
        if self.target in self.feature_cols and self.target not in X_prepared.columns:
            X_prepared[self.target] = np.nan
        for col in self.feature_cols:
            if col not in X_prepared.columns:
                X_prepared[col] = np.nan
        return X_prepared[self.feature_cols]
