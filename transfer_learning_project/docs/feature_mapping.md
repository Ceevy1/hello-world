# 特征对齐说明

- `exam_score -> assessment_score`：MinMax缩放
- `exercise_1~3 -> mean_click_per_activity`：均值后Z-score
- `lab_1~7 -> vle_interaction_t1~t7`：插值后时序对齐
- `report -> tma_score`：MinMax缩放
- `lab_total -> sum_vle_interactions`：Quantile对齐
- `regular_score -> studied_credits`：Rank归一化
- `final_score -> final_result_score`：MinMax缩放
- `exercise_cv -> click_stability`：Standard分布匹配
