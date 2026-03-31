import numpy as np
import pandas as pd


class FeatureEngineering:
    RAW_COLS = {
        'attendance': '考勤',
        'exercise': ['练习1', '练习2', '练习3'],
        'lab': ['实验1', '实验2', '实验3', '实验4', '实验5', '实验6', '实验7'],
        'report': '报告',
        'agg_process': '总平时 成绩',
        'agg_lab': '总实验 成绩',
        'regular': '平时成绩',
        'final_exam': '总期末成绩',
        'target': '总评成绩',
    }

    def _compute_trend(self, series: np.ndarray) -> float:
        x = np.arange(len(series))
        valid = ~np.isnan(series)
        if valid.sum() < 2:
            return 0.0
        return float(np.polyfit(x[valid], series[valid], 1)[0])

    def _compute_cv(self, series: np.ndarray) -> float:
        return float(np.nanstd(series) / (np.nanmean(series) + 1e-6))

    def _add_common_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ex_cols = self.RAW_COLS['exercise']
        lb_cols = self.RAW_COLS['lab']

        out['attendance_rate'] = out[self.RAW_COLS['attendance']] / 100.0
        out['exercise_mean'] = out[ex_cols].mean(axis=1)
        out['exercise_trend'] = out[ex_cols].apply(lambda r: self._compute_trend(r.to_numpy(dtype=float)), axis=1)
        out['exercise_cv'] = out[ex_cols].apply(lambda r: self._compute_cv(r.to_numpy(dtype=float)), axis=1)
        out['lab_mean'] = out[lb_cols].mean(axis=1)
        out['lab_trend'] = out[lb_cols].apply(lambda r: self._compute_trend(r.to_numpy(dtype=float)), axis=1)
        out['lab_cv'] = out[lb_cols].apply(lambda r: self._compute_cv(r.to_numpy(dtype=float)), axis=1)
        out['report_quality'] = out[self.RAW_COLS['report']] / (out[self.RAW_COLS['agg_lab']] + 1e-6)
        out['process_exam_gap'] = out[self.RAW_COLS['regular']] - out[self.RAW_COLS['final_exam']]
        out['early_warning_score'] = 0.3 * out[self.RAW_COLS['attendance']] + 0.4 * out['exercise_mean'] + 0.3 * out['lab_mean']
        return out

    def build_features(self, df: pd.DataFrame, stage: str) -> pd.DataFrame:
        stage = stage.upper()
        d = self._add_common_derived(df)

        t1_cols = ['考勤', '练习1', 'attendance_rate']
        t2_cols = t1_cols + ['练习2', '实验1', '实验2', '实验3']
        t3_cols = t2_cols + ['练习3', '实验4', '实验5', '实验6', '实验7', '报告']

        derived_early = ['exercise_mean', 'exercise_trend', 'exercise_cv', 'early_warning_score']
        derived_full = derived_early + ['lab_mean', 'lab_trend', 'lab_cv', 'report_quality', 'process_exam_gap']

        if stage == 'T1':
            cols = t1_cols + ['exercise_mean', 'exercise_trend', 'early_warning_score']
        elif stage == 'T2':
            cols = t2_cols + derived_early + ['lab_mean', 'lab_trend']
        elif stage == 'T3':
            cols = t3_cols + derived_full
        elif stage == 'T4':
            cols = [v for v in self.RAW_COLS.values() if isinstance(v, str)] + self.RAW_COLS['exercise'] + self.RAW_COLS['lab'] + derived_full
        else:
            raise ValueError(f'Unknown stage: {stage}')

        cols = [c for c in dict.fromkeys(cols) if c in d.columns and c != self.RAW_COLS['target']]
        return d[cols].copy()

    def get_feature_groups(self, stage: str) -> dict:
        f = self.build_features(pd.DataFrame([{  # schema-only mock
            '考勤': 0, '练习1': 0, '练习2': 0, '练习3': 0,
            '实验1': 0, '实验2': 0, '实验3': 0, '实验4': 0, '实验5': 0, '实验6': 0, '实验7': 0,
            '报告': 0, '总平时 成绩': 0, '总实验 成绩': 0, '平时成绩': 0, '总期末成绩': 0, '总评成绩': 0
        }]), stage)
        cols = list(f.columns)
        s1 = [c for c in cols if c.startswith('练习') or c in ['考勤', 'attendance_rate', 'exercise_mean', 'exercise_trend', 'exercise_cv', 'early_warning_score']]
        s2 = [c for c in cols if c.startswith('实验') or c in ['报告', 'lab_mean', 'lab_trend', 'lab_cv', 'report_quality']]
        s3 = [c for c in cols if c in ['总期末成绩', '平时成绩', 'process_exam_gap', '总实验 成绩', '总平时 成绩']]
        return {'source1': s1, 'source2': s2, 'source3': s3}
