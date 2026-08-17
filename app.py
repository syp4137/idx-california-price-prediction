from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "week9_streamlit" / "streamlit_demo_price_pipeline.joblib"


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def format_currency(value):
    return f"${value:,.0f}"


st.set_page_config(
    page_title="California Home Price Predictor",
    layout="centered",
)

st.title("California Home Price Predictor")
st.caption("Streamlit demo model using LivingArea, Beds, Baths, and LotSize.")

try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = artifact["model"]
features = artifact["features"]
metrics = artifact.get("metrics", {})

with st.form("prediction_form"):
    living_area = st.number_input(
        "Living Area (sq ft)",
        min_value=1,
        max_value=15000,
        value=1800,
        step=50,
    )
    beds = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=20,
        value=3,
        step=1,
    )
    baths = st.number_input(
        "Bathrooms",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
    )
    lot_size = st.number_input(
        "Lot Size (sq ft)",
        min_value=0,
        max_value=4_500_000,
        value=7200,
        step=100,
    )

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "LivingArea": living_area,
                "BedroomsTotal": beds,
                "BathroomsTotalInteger": baths,
                "LotSizeSquareFeet": lot_size,
            }
        ]
    )[features]

    predicted_price = model.predict(input_data)[0]
    predicted_price = max(float(predicted_price), 0)

    st.metric("Predicted Price", format_currency(predicted_price))

st.divider()

st.subheader("Demo Model Details")
st.write(
    "This app loads a saved joblib artifact that includes both the trained model "
    "and the preprocessing imputer used during training."
)

if metrics:
    st.write(
        {
            "MAE": format_currency(metrics["mae"]),
            "RMSE": format_currency(metrics["rmse"]),
            "Median absolute percentage error": f"{metrics['mdape']:.1%}",
            "R-squared": f"{metrics['r2']:.3f}",
        }
    )

st.info(
    "This simplified demo uses only four property features. Predictions are less "
    "accurate than the full project model because location and other MLS features "
    "are not included."
)
