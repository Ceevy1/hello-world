import pandas as pd

from src.data.feature_aligner import FeatureAligner


def test_feature_aligner_output_columns():
    df = pd.DataFrame(
        {
            "exam_score": [70, 80],
            "exercise_1": [70, 75],
            "exercise_2": [72, 78],
            "exercise_3": [74, 82],
            "lab_1": [80, 82],
            "lab_2": [81, 83],
            "lab_3": [82, 84],
            "lab_4": [83, 85],
            "lab_5": [84, 86],
            "lab_6": [85, 87],
            "lab_7": [86, 88],
            "report": [88, 90],
            "lab_total": [84, 86],
            "regular_score": [82, 83],
            "final_score": [76, 79],
            "exercise_cv": [0.01, 0.02],
        }
    )
    aligned = FeatureAligner().align(df)
    assert "assessment_score" in aligned.columns
    assert "click_stability" in aligned.columns
