from dataclasses import dataclass


@dataclass(frozen=True)
class StageManager:
    stage_order: tuple = ('T1', 'T2', 'T3', 'T4')

    def validate(self, stage: str) -> str:
        s = stage.upper()
        if s not in self.stage_order:
            raise ValueError(f'Invalid stage: {stage}')
        return s
