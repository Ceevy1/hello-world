from __future__ import annotations



def generate_intervention(counterfactuals: list[tuple[str, float, float]]) -> list[str]:
    recommendations: list[str] = []

    for feature, _old, _new in counterfactuals:
        name = feature.lower()
        if "click" in name or "activity" in name:
            recommendations.append("增加学习访问次数")
        elif "time" in name or "session" in name or "week" in name:
            recommendations.append("延长学习时间")
        elif "correct_rate" in name or "correct" in name or "score" in name:
            recommendations.append("提高作业正确率")
        elif "entropy" in name or "diversity" in name:
            recommendations.append("提升学习行为多样性")
        elif "attempt" in name:
            recommendations.append("减少无效重复尝试并优化答题策略")
        elif "credit" in name:
            recommendations.append("优先完成高权重课程任务")

    return sorted(set(recommendations))
