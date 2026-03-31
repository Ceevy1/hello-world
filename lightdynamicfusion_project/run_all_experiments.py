#!/usr/bin/env python3
import argparse
import importlib
import yaml

from src.utils.seed import set_global_seed

EXPERIMENTS = {
    1: ('实验一：动态时序预测', 'experiments.exp1_dynamic_stages'),
    2: ('实验二：多源消融', 'experiments.exp2_ablation_sources'),
    3: ('实验三：基线对比', 'experiments.exp3_baseline_compare'),
    4: ('实验四：可解释性', 'experiments.exp4_interpretability'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='lightdynamicfusion_project/config.yaml')
    parser.add_argument('--exp', default='all')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_global_seed(cfg['project']['seed'])
    exp_ids = list(EXPERIMENTS.keys()) if args.exp == 'all' else [int(x) for x in args.exp.split(',')]

    for exp_id in exp_ids:
        name, module_path = EXPERIMENTS[exp_id]
        print('=' * 60)
        print(f'运行 {name}...')
        mod = importlib.import_module(module_path)
        mod.run(cfg, debug=args.debug)
        print(f'{name} 完成！')


if __name__ == '__main__':
    main()
