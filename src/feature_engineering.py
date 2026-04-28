import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Transform raw canteen demand data into model-ready features."""

    def __init__(self) -> None:
        pass

    def build_features(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features and encode categorical variables."""
        try:
            logger.info("Building feature set with %s records.", len(data_frame))
            df = data_frame.copy()
            df["HourSin"], df["HourCos"] = self._cyclical_transform(df["Hour"], 24)
            df["DaySin"], df["DayCos"] = self._cyclical_transform(df["DayOfWeek"], 7)
            df = self._encode_categories(df, ["Weather", "MenuType"])
            logger.info("Feature engineering completed.")
            return df
        except Exception as error:
            logger.exception("Feature engineering failed: %s", error)
            raise

    def _encode_categories(self, data_frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = data_frame.copy()
        for column in columns:
            df = pd.get_dummies(df, columns=[column], prefix=column, drop_first=True)
        return df

    def _cyclical_transform(self, series: pd.Series, period: int):
        radians = 2 * 3.141592653589793 * series / period
        return pd.Series(np.sin(radians)), pd.Series(np.cos(radians))
