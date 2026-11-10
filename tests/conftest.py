import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.config import MODEL_PATH


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    if not MODEL_PATH.exists():
        from src.train import train

        train()


@pytest.fixture()
def client():
    with TestClient(app) as c:  # triggers lifespan -> loads model
        yield c


@pytest.fixture()
def strong_student():
    return {"cgpa": 9.2, "iq": 125, "internships": 3, "projects": 5, "certifications": 4,
            "communication_skills": 9, "backlogs": 0, "extracurricular": 1}


@pytest.fixture()
def weak_student():
    return {"cgpa": 5.2, "iq": 80, "internships": 0, "projects": 0, "certifications": 0,
            "communication_skills": 3, "backlogs": 3, "extracurricular": 0}
