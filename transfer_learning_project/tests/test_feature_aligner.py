import pandas as pd

from src.data.feature_aligner import FeatureAligner


def test_feature_aligner_custom_output_columns():
    df = pd.DataFrame(
        {
            "exam_score": [70, 80],
            "exercise_1": [70, 75],
            "exercise_2": [72, 78],
            "exercise_3": [74, 82],
            "report": [88, 90],
            "lab_total": [84, 86],
            "regular_score": [82, 83],
            "final_score": [76, 79],
            "target": [79, 84],
        }
    )
    aligned = FeatureAligner().align(df, domain="custom")
    assert {"performance", "engagement", "behavior", "background", "target"}.issubset(aligned.columns)
