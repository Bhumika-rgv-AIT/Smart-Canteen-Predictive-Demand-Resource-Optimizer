import logging
import os
import sys
from typing import Dict

import pandas as pd
import streamlit as st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_ingestion import SyntheticDataIngestion
from feature_engineering import FeatureEngineering
from model_pipeline import DemandModelPipeline
from resource_optimizer import ResourceOptimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_model_and_data() -> Dict[str, object]:
    """Load synthetic data, feature engineering, and train the predictive pipeline."""
    ingestion = SyntheticDataIngestion()
    data_frame = ingestion.load_data()
    features = FeatureEngineering().build_features(data_frame)
    model_pipeline = DemandModelPipeline()
    evaluation, trained_model = model_pipeline.train(features)
    return {
        "data": data_frame,
        "features": features,
        "model": trained_model,
        "evaluation": evaluation,
    }


def build_input_row(exam_period: bool, menu_type: str, weather: str, hour: int, day_of_week: int) -> pd.DataFrame:
    """Construct a model input row for user scenario simulation."""
    base = {
        "DayOfWeek": day_of_week,
        "Hour": hour,
        "Holiday": int(day_of_week >= 5),
        "ExamPeriod": int(exam_period),
        "Weather_Rainy": int(weather == "Rainy"),
        "Weather_Stormy": int(weather == "Stormy"),
        "Weather_Sunny": int(weather == "Sunny"),
        "MenuType_Regular": int(menu_type == "Regular"),
        "MenuType_Vegetarian": int(menu_type == "Vegetarian"),
    }
    row = pd.DataFrame([base])
    row["HourSin"], row["HourCos"] = FeatureEngineering()._cyclical_transform(row["Hour"], 24)
    row["DaySin"], row["DayCos"] = FeatureEngineering()._cyclical_transform(row["DayOfWeek"], 7)
    return row


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(page_title="Smart Canteen Demand Optimizer", layout="wide")
    st.title("Smart Canteen Demand & Resource Optimizer")
    st.markdown(
        "Use the controls to project demand changes for exam periods, menu selections, and weather conditions."
    )

    with st.spinner("Loading synthetic data and training the model..."):
        context = load_model_and_data()

    evaluation = context["evaluation"]
    st.sidebar.header("Simulation Controls")
    exam_period = st.sidebar.checkbox("Exam Period", value=True)
    menu_type = st.sidebar.selectbox("Menu Type", ["Regular", "Premium", "Vegetarian"])
    weather = st.sidebar.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Stormy"])
    hour = st.sidebar.slider("Hour of Day", min_value=7, max_value=18, value=12)
    day_of_week = st.sidebar.selectbox(
        "Day of Week", list(range(0, 7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x]
    )

    input_row = build_input_row(exam_period, menu_type, weather, hour, day_of_week)
    prediction = int(context["model"].predict(input_row)[0])
    optimizer = ResourceOptimizer()
    plan = optimizer.create_plan(prediction, evaluation["baseline_RMSE"])

    st.subheader("Predicted Footfall Scenario")
    st.metric("Predicted Footfall", int(plan["PredictedFootfall"]))
    st.metric("Estimated Staff Required", int(plan["StaffRequired"]))

    st.subheader("Resource Planning Summary")
    st.write(
        {
            "Inventory Budget (INR)": plan["InventoryBudget"],
            "Energy Usage (kWh)": plan["EnergyUsage_kWh"],
            "Potential Waste Savings (INR)": plan["WasteSavingsINR"],
        }
    )

    st.sidebar.header("Model Comparison")
    st.sidebar.write(
        {
            "Baseline R2": round(evaluation["baseline_R2"], 3),
            "Baseline RMSE": round(evaluation["baseline_RMSE"], 2),
            "Tuned R2": round(evaluation["tuned_R2"], 3),
            "Tuned RMSE": round(evaluation["tuned_RMSE"], 2),
        }
    )

    st.subheader("Dataset Sample")
    st.dataframe(context["data"].head(10))


if __name__ == "__main__":
    main()
