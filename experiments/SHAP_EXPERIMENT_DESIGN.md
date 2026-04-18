# SHAP 实验设计（补充到 /experiments）

> 目标：为论文补齐“可解释性”实验，覆盖**全局解释、单样本解释、融合权重动态解释**三层。

## 1) 解释对象分三层

### ✅ (1) Tree 模型全局解释（最重要）

- **模型对象**：`XGBoost`、`CatBoost`
- **输入特征**：`tab_features`（统计行为、成绩、活跃度、时间等结构化特征）
- **方法**：
  - 训练后分别计算 SHAP values
  - 统一映射到同一特征名空间，便于跨模型对比
- **输出（论文图/表）**：
  1. `SHAP summary plot`（全局）
  2. `feature ranking`（Top-k 特征重要性表）

**建议产物路径**
- `outputs/shap/xgb_shap_values.npy`
- `outputs/shap/cat_shap_values.npy`
- `outputs/shap/feature_ranking.csv`
- `figures/shap_summary_xgb.png`
- `figures/shap_summary_cat.png`

---

### ✅ (2) 单样本解释（Case Study）

- **对象**：高风险样本（预测 FAIL）与低风险样本（预测 PASS）各选 1~3 个
- **解释模板**（论文可直接引用）：

> Student A predicted **FAIL** because:
> - low activity (`-0.31`)
> - low score (`-0.42`)

- **方法**：
  - 对目标样本计算局部 SHAP
  - 输出贡献前 5 个正/负特征
  - 采用 `force plot` + 文本解释双呈现
- **输出（论文图）**：
  1. `SHAP force plot`（单样本）
  2. `local contribution table`

**建议产物路径**
- `outputs/shap/local_case_student_a.csv`
- `figures/shap_force_student_a.png`

---

### ✅ (3) Fusion 权重分析（创新点）

- **核心问题**：动态融合模型在不同学习阶段更依赖哪类信息？
- **统计对象**：按 week 聚合的融合权重（tree 分支 vs transformer 分支）
- **假设与预期**：
  - `early stage` → more tree（结构化先验更稳定）
  - `late stage` → more transformer（序列行为信息更充分）
- **方法**：
  - 记录每周样本的 fusion gate / attention 权重
  - 对周维度做均值与置信区间统计
- **输出（论文图）**：
  1. `Fusion weight vs week` 折线图（tree/transformer 双曲线）

**建议产物路径**
- `outputs/shap/fusion_weight_by_week.csv`
- `figures/fusion_weight_vs_week.png`

---

## 2) 论文必备可视化（建议 3 张图）

1. **SHAP summary plot（全局）**
2. **SHAP force plot（单样本）**
3. **Fusion weight vs week（折线图）**

> 若版面允许，可补一张 `Top-10 feature ranking` 柱状图增强可读性。

---

## 3) 实验执行流程（可直接落地）

1. 训练并固化 `XGB/CAT` 最优模型（固定 seed）
2. 计算全局 SHAP + 导出 ranking
3. 选取 FAIL/PASS 代表样本做局部解释
4. 汇总 fusion 权重周变化并绘图
5. 统一输出到 `outputs/shap/` 和 `figures/`

---

## 4) 报告指标与一致性检查

- **稳定性**：不同 seed 下 Top-10 交并比（Jaccard）
- **一致性**：XGB vs CAT 排名相关系数（Spearman）
- **可读性**：单样本解释是否与真实学习行为一致（专家复核）

