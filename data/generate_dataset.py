"""Generate a reproducible synthetic student placement dataset.

Run:  python data/generate_dataset.py
Output: data/student_placement.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 1500


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cgpa = np.clip(rng.normal(7.4, 1.0, n), 4.5, 10).round(2)
    iq = np.clip(rng.normal(100, 15, n), 60, 150).round(0).astype(int)
    internships = rng.choice([0, 1, 2, 3], n, p=[0.35, 0.35, 0.22, 0.08])
    projects = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.08, 0.2, 0.28, 0.24, 0.14, 0.06])
    certifications = rng.choice([0, 1, 2, 3, 4], n, p=[0.25, 0.3, 0.25, 0.14, 0.06])
    communication_skills = np.clip(rng.normal(6.0, 1.8, n), 1, 10).round(1)
    backlogs = rng.choice([0, 1, 2, 3], n, p=[0.72, 0.17, 0.08, 0.03])
    extracurricular = rng.choice([0, 1], n, p=[0.45, 0.55])

    # Hidden "ground truth" rule + noise -> realistic, learnable but imperfect
    z = (
        -11.0
        + 0.9 * cgpa
        + 0.03 * (iq - 100)
        + 0.55 * internships
        + 0.30 * projects
        + 0.25 * certifications
        + 0.28 * communication_skills
        - 0.9 * backlogs
        + 0.3 * extracurricular
        + rng.normal(0, 0.9, n)
    )
    placed = (1 / (1 + np.exp(-z)) > 0.5).astype(int)

    return pd.DataFrame(
        {
            "cgpa": cgpa,
            "iq": iq,
            "internships": internships,
            "projects": projects,
            "certifications": certifications,
            "communication_skills": communication_skills,
            "backlogs": backlogs,
            "extracurricular": extracurricular,
            "placed": placed,
        }
    )


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "student_placement.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}  | placement rate = {df.placed.mean():.2%}")
