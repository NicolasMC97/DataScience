from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class PredictResponse(BaseModel):
    label: str
    probability: float | None = None  # if available

class HealthResponse(BaseModel):
    status: str
