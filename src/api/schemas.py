"""
Pydantic request/response schemas for the Predictive Maintenance API.
"""

from typing import Literal
from pydantic import BaseModel, Field, model_validator


class MachineReading(BaseModel):
    """
    Single sensor reading from an industrial machine.
    Mirrors the AI4I 2020 feature set.
    """
    type: Literal["L", "M", "H"] = Field(
        ...,
        description="Machine quality variant: L=Low, M=Medium, H=High",
        alias="Type",
    )
    air_temperature_k: float = Field(
        ..., ge=295.0, le=310.0,
        description="Air temperature in Kelvin",
        alias="Air temperature [K]",
    )
    process_temperature_k: float = Field(
        ..., ge=305.0, le=315.0,
        description="Process temperature in Kelvin",
        alias="Process temperature [K]",
    )
    rotational_speed_rpm: int = Field(
        ..., ge=1000, le=3000,
        description="Rotational speed in RPM",
        alias="Rotational speed [rpm]",
    )
    torque_nm: float = Field(
        ..., ge=0.0, le=100.0,
        description="Torque in Newton-meters",
        alias="Torque [Nm]",
    )
    tool_wear_min: int = Field(
        ..., ge=0, le=300,
        description="Tool wear accumulation in minutes",
        alias="Tool wear [min]",
    )

    model_config = {"populate_by_name": True}

    def to_raw_dict(self) -> dict:
        """Return dict with original AI4I column names for the preprocessor."""
        return {
            "Type": self.type,
            "Air temperature [K]": self.air_temperature_k,
            "Process temperature [K]": self.process_temperature_k,
            "Rotational speed [rpm]": self.rotational_speed_rpm,
            "Torque [Nm]": self.torque_nm,
            "Tool wear [min]": self.tool_wear_min,
        }


class PredictionResponse(BaseModel):
    failure_predicted: bool
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    threshold_used: float
    model_version: str


class ExplainResponse(PredictionResponse):
    explanation: dict


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None
    version: str
