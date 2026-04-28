import logging
from math import ceil
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class ResourceOptimizer:
    """Map predicted footfall values into staffing, inventory, and energy plans."""

    def __init__(self, average_plate_cost: float = 75.0) -> None:
        """Initialize the optimizer with an India-specific average plate cost in INR."""
        self.average_plate_cost = average_plate_cost

    def create_plan(self, predicted_footfall: int, baseline_rmse: float) -> Dict[str, float]:
        """Create a resource plan for a predicted footfall value."""
        try:
            staff_required = ceil(predicted_footfall / 50)
            energy_usage = self._estimate_energy(predicted_footfall)
            waste_savings = self._estimate_waste_savings(predicted_footfall, baseline_rmse)
            plan = {
                "PredictedFootfall": float(predicted_footfall),
                "StaffRequired": staff_required,
                "InventoryBudget": float(predicted_footfall * self.average_plate_cost),
                "EnergyUsage_kWh": energy_usage,
                "WasteSavingsINR": waste_savings,
            }
            logger.info("Resource plan created for predicted footfall %s.", predicted_footfall)
            return plan
        except Exception as error:
            logger.exception("Resource planning failed: %s", error)
            raise

    def _estimate_energy(self, predicted_footfall: int) -> float:
        return round(predicted_footfall * 0.18, 2)

    def _estimate_waste_savings(self, predicted_footfall: int, baseline_rmse: float) -> float:
        error_delta = baseline_rmse * 0.10
        return float(max(0.0, predicted_footfall * error_delta * 0.5))
