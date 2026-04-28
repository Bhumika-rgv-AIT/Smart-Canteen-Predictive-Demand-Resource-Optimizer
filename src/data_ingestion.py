import logging
from datetime import datetime, timedelta
from math import ceil
from random import choice, gauss, randint
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

WEATHER_CATEGORIES = ["Sunny", "Cloudy", "Rainy", "Stormy"]
MENU_TYPES = ["Regular", "Premium", "Vegetarian"]


class SyntheticDataIngestion:
    """Generate synthetic canteen demand data with seasonality, noise, and business signals."""

    def __init__(self, start_date: str = "2025-01-01", periods: int = 520) -> None:
        self.start_date = datetime.fromisoformat(start_date)
        self.periods = periods

    def load_data(self) -> pd.DataFrame:
        """Build a synthetic dataset with realistic footfall and context features."""
        try:
            logger.info("Starting synthetic dataset generation for %s records.", self.periods)
            records = []
            for day_offset in range(self.periods):
                record_date = self.start_date + timedelta(days=day_offset)
                weekday = record_date.weekday()
                hour = randint(7, 18)
                holiday = self._is_holiday(record_date)
                weather = choice(WEATHER_CATEGORIES)
                menu_type = choice(MENU_TYPES)
                exam_period = self._is_exam_period(record_date)
                base_footfall = self._daily_base(hour, weekday, holiday)
                weather_adjustment = self._weather_impact(weather)
                menu_adjustment = self._menu_impact(menu_type)
                exam_adjustment = 1.15 if exam_period else 1.0
                noise = gauss(0, 12)
                predicted = max(10, int(base_footfall * weather_adjustment * menu_adjustment * exam_adjustment + noise))
                records.append(
                    {
                        "Date": record_date,
                        "DayOfWeek": weekday,
                        "Hour": hour,
                        "Holiday": int(holiday),
                        "Weather": weather,
                        "MenuType": menu_type,
                        "ExamPeriod": int(exam_period),
                        "Footfall": predicted,
                    }
                )
            data_frame = pd.DataFrame(records)
            logger.info("Synthetic dataset generation completed successfully.")
            return data_frame
        except Exception as error:
            logger.exception("Failed to generate synthetic data: %s", error)
            raise

    def _is_holiday(self, date: datetime) -> bool:
        return date.weekday() in {5, 6} or date.day in {1, 15}

    def _is_exam_period(self, date: datetime) -> bool:
        return 1 <= date.month <= 4 and date.weekday() < 5

    def _daily_base(self, hour: int, weekday: int, holiday: bool) -> float:
        base = 80 if hour in {11, 12, 13} else 35
        weekend_multiplier = 0.75 if weekday >= 5 else 1.0
        holiday_multiplier = 0.7 if holiday else 1.0
        return base * weekend_multiplier * holiday_multiplier

    def _weather_impact(self, weather: str) -> float:
        impacts = {"Sunny": 1.0, "Cloudy": 0.95, "Rainy": 0.85, "Stormy": 0.7}
        return impacts.get(weather, 1.0)

    def _menu_impact(self, menu_type: str) -> float:
        impacts = {"Regular": 1.0, "Premium": 1.12, "Vegetarian": 0.95}
        return impacts.get(menu_type, 1.0)
