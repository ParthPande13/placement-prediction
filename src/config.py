from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "student_placement.csv"
MODEL_PATH = ROOT / "models" / "placement_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURES = [
    "cgpa",
    "iq",
    "internships",
    "projects",
    "certifications",
    "communication_skills",
    "backlogs",
    "extracurricular",
]
TARGET = "placed"
RANDOM_STATE = 42
