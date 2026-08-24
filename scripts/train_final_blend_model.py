from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.final_blend_model import (
    FinalBlendPriceModel,
    TARGET,
    build_local_aggregate_xgboost_pipeline,
    transform_target_for_mode,
)


BEST_XGB_PARAMS = {
    "max_depth": 9,
    "learning_rate": 0.10,
    "n_estimators": 1200,
}

CATEGORICAL_DTYPE_COLS = [
    "City",
    "CountyOrParish",
    "PostalCode",
    "MLSAreaMajor",
    "Levels",
    "HighSchoolDistrict",
    "UnifiedSchoolDistrict",
]

LOCATION_QUALITY_FLAG_COLS = [
    "Latitude_missing",
    "Longitude_missing",
    "invalid_coordinates_flag",
    "PostalCode_format_issue_flag",
]

BASE_NUMERIC_COLS = [
    "LivingArea",
    "LotSizeSquareFeet",
    "Latitude",
    "Longitude",
    "AssociationFee",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "GarageSpaces",
    "ParkingTotal",
    "Stories",
    "Flooring_material_count",
]

STRUCTURAL_MODEL_NUMERIC_COLS = [
    "PropertyAgeYears",
    "BedBathRatio",
    "LivingAreaPerBedroom",
    "LogLotToLivingAreaRatio",
]

BASE_BOOLEAN_COLS = [
    "ViewYN",
    "PoolPrivateYN",
    "AttachedGarageYN",
    "FireplaceYN",
    "NewConstructionYN",
    "Flooring_Carpet",
    "Flooring_Tile",
    "Flooring_Wood",
    "Flooring_Laminate",
    "Flooring_Vinyl",
    "Flooring_Stone",
    "Flooring_Concrete",
    "Flooring_Bamboo",
    "Flooring_Brick",
    "Flooring_SeeRemarks",
    "Flooring_multiple_materials_flag",
]

BASE_CATEGORICAL_COLS = [
    "City",
    "CountyOrParish",
    "PostalCode",
    "MLSAreaMajor",
    "Levels",
    "HighSchoolDistrict",
    "UnifiedSchoolDistrict",
]


def find_project_root() -> Path:
    candidate_roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd() / "idx-california-price-prediction",
    ]
    return next(
        root
        for root in candidate_roots
        if (root / "data" / "processed").exists() and (root / "notebooks").exists()
    )


def normalize_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_DTYPE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("__missing__")
    return df


def load_split_file(path: Path) -> pd.DataFrame:
    df = normalize_categorical_dtypes(pd.read_csv(path, low_memory=False))
    if "CloseDate" in df.columns:
        df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")
    if "close_month" in df.columns:
        df["close_month"] = pd.PeriodIndex(df["close_month"], freq="M")
    return df


def available(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def compute_price_per_sqft(
    df: pd.DataFrame,
    target_col: str = TARGET,
    living_area_col: str = "LivingArea",
) -> pd.Series:
    close_price = pd.to_numeric(df[target_col], errors="coerce")
    living_area = pd.to_numeric(df[living_area_col], errors="coerce")
    return close_price.where(close_price.gt(0)) / living_area.where(living_area.gt(0))


def apply_train_price_bounds(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
    ppsf_lower_q: float = 0.01,
    ppsf_upper_q: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    lower_bound = train_df[TARGET].quantile(lower_q)
    upper_bound = train_df[TARGET].quantile(upper_q)
    train_ppsf = compute_price_per_sqft(train_df)
    eval_ppsf = compute_price_per_sqft(eval_df)
    ppsf_lower_bound = train_ppsf.quantile(ppsf_lower_q)
    ppsf_upper_bound = train_ppsf.quantile(ppsf_upper_q)

    train_price_mask = train_df[TARGET].between(lower_bound, upper_bound)
    eval_price_mask = eval_df[TARGET].between(lower_bound, upper_bound)
    train_ppsf_mask = train_ppsf.isna() | train_ppsf.between(ppsf_lower_bound, ppsf_upper_bound)
    eval_ppsf_mask = eval_ppsf.isna() | eval_ppsf.between(ppsf_lower_bound, ppsf_upper_bound)
    train_mask = train_price_mask & train_ppsf_mask
    eval_mask = eval_price_mask & eval_ppsf_mask

    return train_df.loc[train_mask].copy(), eval_df.loc[eval_mask].copy(), {
        "price_lower_bound": lower_bound,
        "price_upper_bound": upper_bound,
        "price_per_sqft_lower_bound": ppsf_lower_bound,
        "price_per_sqft_upper_bound": ppsf_upper_bound,
        "train_rows_before_filter": len(train_df),
        "train_rows_after_filter": int(train_mask.sum()),
        "eval_rows_before_filter": len(eval_df),
        "eval_rows_after_filter": int(eval_mask.sum()),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred, index=y_true.index).astype(float)
    ape = ((y_true - y_pred).abs() / y_true).replace([np.inf, -np.inf], np.nan) * 100
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": ape.mean(),
        "mdape": ape.median(),
    }


def build_feature_groups(train_df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    numeric_cols = available(BASE_NUMERIC_COLS + STRUCTURAL_MODEL_NUMERIC_COLS, train_df)
    categorical_cols = available(BASE_CATEGORICAL_COLS, train_df)
    boolean_cols = available(BASE_BOOLEAN_COLS, train_df)
    flag_cols = [
        col
        for col in train_df.columns
        if (col.endswith("_missing") or col == "invalid_coordinates_flag")
        and col not in LOCATION_QUALITY_FLAG_COLS
    ] + available(["HasGarage"], train_df)
    local_source_features = available(["CloseDate", TARGET], train_df)
    feature_cols = list(
        dict.fromkeys(numeric_cols + categorical_cols + boolean_cols + flag_cols + local_source_features)
    )
    return numeric_cols, categorical_cols, boolean_cols, flag_cols, feature_cols


def fit_target_mode(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    boolean_cols: list[str],
    flag_cols: list[str],
    target_transform: str,
):
    model = build_local_aggregate_xgboost_pipeline(
        numeric_cols,
        categorical_cols,
        boolean_cols,
        flag_cols,
        BEST_XGB_PARAMS,
    )
    start_time = time.perf_counter()
    model.fit(train_df[feature_cols], transform_target_for_mode(train_df[TARGET], target_transform))
    fit_seconds = time.perf_counter() - start_time
    return model, fit_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final local + 60/40 blend model artifact.")
    parser.add_argument(
        "--output",
        default="models/final/final_local_60_40_blend_artifact.joblib",
        help="Output joblib path relative to the project root.",
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/final_blend_artifact/final_blend_artifact_metrics.csv",
        help="Output metrics CSV path relative to the project root.",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    split_dir = project_root / "data" / "processed" / "week6_feature_engineering" / "splits"
    split_plan_path = project_root / "data" / "processed" / "crmls_week3_split_plan.csv"
    final_train_path = split_dir / "final_train_fixed_window_engineered.csv"
    final_test_path = split_dir / "test_latest_month_engineered.csv"
    output_path = project_root / args.output
    metrics_output_path = project_root / args.metrics_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    split_plan = pd.read_csv(split_plan_path, dtype={"split_type": "string", "eval_month": "string"})
    final_plan = split_plan[split_plan["split_type"] == "final_test"].copy().iloc[0]
    final_train_raw = load_split_file(final_train_path)
    final_test_raw = load_split_file(final_test_path)
    final_train_df, final_test_df, bound_info = apply_train_price_bounds(final_train_raw, final_test_raw)

    numeric_cols, categorical_cols, boolean_cols, flag_cols, feature_cols = build_feature_groups(final_train_df)

    log1p_model, fit_seconds_log1p = fit_target_mode(
        final_train_df,
        feature_cols,
        numeric_cols,
        categorical_cols,
        boolean_cols,
        flag_cols,
        "log1p",
    )
    raw_model, fit_seconds_raw = fit_target_mode(
        final_train_df,
        feature_cols,
        numeric_cols,
        categorical_cols,
        boolean_cols,
        flag_cols,
        "raw",
    )

    artifact = FinalBlendPriceModel(
        log1p_model=log1p_model,
        raw_model=raw_model,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        boolean_cols=boolean_cols,
        flag_cols=flag_cols,
        log1p_weight=0.6,
        raw_weight=0.4,
        metadata={
            "model_name": "final_local_60_40_blend",
            "feature_set": "Base + Structural Derived + UnifiedSchoolDistrict + local price features",
            "target_prediction": "60% log1p + 40% raw blend",
            "xgboost_params": BEST_XGB_PARAMS,
            "xgboost_version": xgb.__version__,
            "train_month_start": str(final_plan["train_month_start"]),
            "train_month_end": str(final_plan["train_month_end"]),
            "test_month": str(final_plan["eval_month"]),
            "outlier_filter_bounds": bound_info,
            "fit_seconds_log1p": fit_seconds_log1p,
            "fit_seconds_raw": fit_seconds_raw,
        },
    )

    train_predictions = artifact.predict(final_train_df)
    test_predictions = artifact.predict(final_test_df)
    artifact.metrics = {
        "train": regression_metrics(final_train_df[TARGET], train_predictions),
        "test": regression_metrics(final_test_df[TARGET], test_predictions),
    }

    joblib.dump(artifact, output_path)
    metrics_row = {
        "model_name": artifact.metadata["model_name"],
        "feature_set": artifact.metadata["feature_set"],
        "target_prediction": artifact.metadata["target_prediction"],
        "log1p_weight": artifact.log1p_weight,
        "raw_weight": artifact.raw_weight,
        "test_month": artifact.metadata["test_month"],
        "n_raw_features": len(artifact.feature_cols),
        **{f"train_{key}": value for key, value in artifact.metrics["train"].items()},
        **{f"test_{key}": value for key, value in artifact.metrics["test"].items()},
        **bound_info,
        "artifact_path": str(output_path.relative_to(project_root)),
    }
    pd.DataFrame([metrics_row]).to_csv(metrics_output_path, index=False)

    print(f"Saved final blend model artifact: {output_path}")
    print(f"Saved metrics: {metrics_output_path}")
    print(f"Test R2: {artifact.metrics['test']['r2']:.6f}")
    print(f"Test MAE: ${artifact.metrics['test']['mae']:,.2f}")
    print(f"Test RMSE: ${artifact.metrics['test']['rmse']:,.2f}")
    print(f"Test MdAPE: {artifact.metrics['test']['mdape']:.6f}%")


if __name__ == "__main__":
    main()
