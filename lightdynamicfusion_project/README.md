# LightDynamicFusion-S (Small-sample redesign)

该工程实现“架构迁移而非权重迁移”的小样本实验框架，包含：
- 分阶段动态预测（T1~T4）
- 多源分组建模 + 注意力融合
- 小样本CV评估和基线比较

## 快速开始
```bash
python lightdynamicfusion_project/run_all_experiments.py --exp all
```

默认数据使用仓库已有 `data/student_scores.csv`。
