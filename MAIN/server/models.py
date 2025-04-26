from pydantic import BaseModel, Field

class filterData(BaseModel):
    filter: list[list[float]] = Field(..., description="Filter coefficients")