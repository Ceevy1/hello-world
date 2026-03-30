# OULAD → 自建数据迁移学习实验框架

## 快速开始

```bash
cd transfer_learning_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --config configs/config.yaml --phase 0-5
```

## 目录
- `src/data`: 数据加载、特征工程、特征对齐
- `src/models`: 预训练模型加载、域适应、迁移模型
- `src/training`: 训练与微调策略
- `src/evaluation`: 指标与可视化
- `experiments`: 各Phase运行脚本

## 说明
- 默认不包含任何网络请求，满足本地数据处理约束。
- 当 `student_id` 列存在时支持匿名化哈希。
- 输出文件保存到 `results/` 目录。
