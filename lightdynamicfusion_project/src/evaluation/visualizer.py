from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ExperimentVisualizer:
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        sns.set_style('whitegrid')

    def plot_dynamic_stage_performance(self, results: dict, baselines: dict, save_path: str):
        stages = ['T1', 'T2', 'T3', 'T4']
        mae = [results[s]['mae']['mean'] for s in stages]
        mae_std = [results[s]['mae']['std'] for s in stages]

        plt.figure(figsize=(8, 4))
        plt.plot(stages, mae, marker='o', label='LightDynamicFusion-S MAE')
        plt.fill_between(stages, [m-s for m, s in zip(mae, mae_std)], [m+s for m, s in zip(mae, mae_std)], alpha=0.2)
        for b, v in baselines.items():
            plt.axhline(v, linestyle='--', label=f'{b} (T4)')
        plt.legend()
        plt.title('Dynamic Stage Performance')
        plt.ylabel('MAE')
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_attention_heatmap(self, attention_weights: dict, save_path: str):
        df = pd.DataFrame(attention_weights).T[['source1', 'source2', 'source3']]
        plt.figure(figsize=(6, 4))
        sns.heatmap(df, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Attention Weights by Stage')
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
