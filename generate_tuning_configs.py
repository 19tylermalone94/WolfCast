import argparse
import random
import yaml
import os


def load_base_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_random_configs(base_config, num_trials):
    
    param_ranges = {
        'max_depth': [6, 8, 10, 12],
        'eta': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0]
    }
    
    configs = []
    for i in range(num_trials):
        config = yaml.safe_load(yaml.dump(base_config))
        
        xgb_config = config['xgboost']
        for param, values in param_ranges.items():
            xgb_config[param] = random.choice(values)
        
        xgb_config['reg_alpha'] = 0.0
        
        trial_name = f"trial_{i+1:03d}"
        base_name = config['app_name']
        config['app_name'] = f"{base_name}_tuning_{trial_name}"
        
        base_path = config['model_output_path']
        config['model_output_path'] = f"{base_path}_tuning_{trial_name}"
        
        configs.append({
            'trial_id': i + 1,
            'trial_name': trial_name,
            'config': config,
            'hyperparameters': {k: xgb_config[k] for k in param_ranges.keys()}
        })
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='Generate random hyperparameter tuning configs')
    parser.add_argument('--base_config', type=str, required=True,
                       help='Base config file to use (e.g., 54m_1neg_1pos_v1.yaml)')
    parser.add_argument('--num_trials', type=int, default=20,
                       help='Number of random configs to generate (default: 20)')
    parser.add_argument('--output_dir', type=str, default='tuning_configs',
                       help='Directory to save configs (default: tuning_configs)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    base_config = load_base_config(args.base_config)
    
    print(f"Generating {args.num_trials} random hyperparameter combinations...")
    configs = generate_random_configs(base_config, args.num_trials)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nSaving configs to {args.output_dir}/")
    for config_dict in configs:
        filename = f"{config_dict['trial_name']}.yaml"
        filepath = os.path.join(args.output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict['config'], f, default_flow_style=False, sort_keys=False)
        
        print(f"  {filename}")
    

if __name__ == "__main__":
    main()

