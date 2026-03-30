"""Phase 3: transfer model finetuning."""

from __future__ import annotations

import joblib

from src.models.base_model import PretrainedModelLoader
from src.models.domain_adapter import CORALAdapter
from src.models.transfer_model import TransferLearningModel


def run_phase3(config: dict, logger) -> None:
    df = joblib.load("data/processed/target_features.pkl")
    y = df["target"].values
    X = df.drop(columns=["target"]).select_dtypes("number").values

    pretrained = PretrainedModelLoader(config["transfer"]["pretrained_model_path"]).load()
    model = TransferLearningModel(
        pretrained_model=pretrained,
        transfer_strategy="coral+finetune",
        domain_adapter=CORALAdapter(lambda_coral=config["domain_adaptation"]["coral"]["lambda_coral"]),
        finetune_strategy="FT-1",
        task_type="regression",
    )
    model.fit(X, y, X_source=X)
    joblib.dump(model, "results/models/finetuned_model.pkl")
    logger.info("Phase3 complete: finetuned_model.pkl generated")
