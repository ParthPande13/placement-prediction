from typing import List

from pydantic import BaseModel, ConfigDict, Field


class StudentFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cgpa": 8.2,
                "iq": 110,
                "internships": 2,
                "projects": 3,
                "certifications": 2,
                "communication_skills": 7.5,
                "backlogs": 0,
                "extracurricular": 1,
            }
        }
    )

    cgpa: float = Field(..., ge=0, le=10, description="Cumulative GPA (0-10)")
    iq: int = Field(..., ge=50, le=200, description="IQ score")
    internships: int = Field(..., ge=0, le=10, description="Number of internships")
    projects: int = Field(..., ge=0, le=20, description="Number of projects")
    certifications: int = Field(..., ge=0, le=20, description="Number of certifications")
    communication_skills: float = Field(..., ge=0, le=10, description="Communication score (0-10)")
    backlogs: int = Field(..., ge=0, le=20, description="Active backlogs")
    extracurricular: int = Field(..., ge=0, le=1, description="1 if active in extracurriculars, else 0")


class PredictionResponse(BaseModel):
    placed: bool
    prediction: str
    placement_probability: float


class BatchRequest(BaseModel):
    students: List[StudentFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]
