import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


class DemandModelPipeline:
    """Build and evaluate a decision tree regression pipeline for canteen demand forecasting."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: DecisionTreeRegressor | None = None
        self.grid_search: GridSearchCV | None = None

    def build_pipeline(self, features: pd.DataFrame) -> Pipeline:
        """Create a Scikit-Learn pipeline with feature scaling and a decision tree regressor."""
        numeric_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
        transformer = ColumnTransformer([
            ("scaler", StandardScaler(), numeric_features)
        ], remainder="passthrough")
        pipeline = Pipeline([
            ("transform", transformer),
            ("regressor", DecisionTreeRegressor(random_state=self.random_state))
        ])
        return pipeline

    def train(self, data_frame: pd.DataFrame) -> Tuple[Dict[str, float], Pipeline]:
        """Train baseline and tuned models, returning evaluation metrics and the tuned pipeline."""
        try:
            logger.info("Preparing training and validation data.")
            target = data_frame["Footfall"]
            features = data_frame.drop(columns=["Date", "Footfall"])
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=self.random_state
            )
            baseline_pipeline = self.build_pipeline(features)
            baseline_pipeline.fit(X_train, y_train)
            y_baseline = baseline_pipeline.predict(X_test)
            baseline_metrics = self._evaluate_metrics(y_test, y_baseline)
            logger.info("Baseline model R2: %.3f, RMSE: %.3f", baseline_metrics["R2"], baseline_metrics["RMSE"])
            param_grid = {
                "regressor__max_depth": [4, 6, 8, 10],
                "regressor__min_samples_leaf": [5, 10, 20],
                "regressor__min_samples_split": [8, 12, 20],
            }
            self.grid_search = GridSearchCV(
                baseline_pipeline,
                param_grid=param_grid,
                cv=5,
                scoring="r2",
                n_jobs=-1,
                verbose=0,
            )
            self.grid_search.fit(X_train, y_train)
            self.model = self.grid_search.best_estimator_
            y_tuned = self.model.predict(X_test)
            tuned_metrics = self._evaluate_metrics(y_test, y_tuned)
            logger.info("Tuned model best params: %s", self.grid_search.best_params_)
            logger.info("Tuned model R2: %.3f, RMSE: %.3f", tuned_metrics["R2"], tuned_metrics["RMSE"])
            return {
                "baseline_R2": baseline_metrics["R2"],
                "baseline_RMSE": baseline_metrics["RMSE"],
                "tuned_R2": tuned_metrics["R2"],
                "tuned_RMSE": tuned_metrics["RMSE"],
            }, self.model
        except Exception as error:
            logger.exception("Model training failed: %s", error)
            raise

    def _evaluate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        rmse_value = mean_squared_error(y_true, y_pred)
        return {
            "R2": float(r2_score(y_true, y_pred)),
            "RMSE": float(rmse_value ** 0.5),
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict footfall values using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(features)
