import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_ingestion import SyntheticDataIngestion
from feature_engineering import FeatureEngineering
from model_pipeline import DemandModelPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def create_plots(output_directory: str = "plots") -> None:
    """Generate professional visualizations for footfall prediction and model residuals."""
    try:
        output_path = Path(output_directory)
        output_path.mkdir(exist_ok=True)
        data_frame = SyntheticDataIngestion().load_data()
        features = FeatureEngineering().build_features(data_frame)
        pipeline = DemandModelPipeline()
        _, trained_model = pipeline.train(features)
        X = features.drop(columns=["Date", "Footfall"])
        y_true = features["Footfall"]
        y_pred = trained_model.predict(X)

        sns.set_theme(style="darkgrid")
        plot_heatmap(data_frame, output_path)
        plot_residuals(y_true, y_pred, output_path)
        logger.info("Visualizations created in %s.", output_path)
    except Exception as error:
        logger.exception("Visualization generation failed: %s", error)
        raise


def plot_heatmap(data_frame, output_path: Path) -> None:
    """Create a heatmap showing predicted footfall by hour and day of week."""
    pivot = (
        data_frame.groupby(["DayOfWeek", "Hour"])["Footfall"]
        .mean()
        .reset_index()
        .pivot(index="DayOfWeek", columns="Hour", values="Footfall")
    )
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
    plt.title("Average Predicted Footfall by Hour and Day of Week")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    heatmap_file = output_path / "footfall_heatmap.png"
    plt.savefig(heatmap_file, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, output_path: Path) -> None:
    """Create a residual plot to visualize model error distribution."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.55)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residual Plot for Footfall Prediction")
    plt.xlabel("Predicted Footfall")
    plt.ylabel("Residuals")
    residuals_file = output_path / "footfall_residuals.png"
    plt.savefig(residuals_file, bbox_inches="tight")
    plt.close()
