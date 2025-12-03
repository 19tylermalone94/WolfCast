import argparse
import json
import os
import glob
import pandas as pd

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


def load_metrics_from_gcs(gs_path):
    """Load all metrics JSON files from GCS."""
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage required for GCS paths")
    
    client = storage.Client()
    bucket_name = gs_path.split("/")[2]
    prefix = "/".join(gs_path.split("/")[3:])
    if not prefix.endswith("/"):
        prefix += "/"
    
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    metrics = []
    for blob in blobs:
        if blob.name.endswith('_metrics.json'):
            try:
                content = blob.download_as_text()
                metric = json.loads(content)
                metrics.append(metric)
            except Exception as e:
                print(f"Error loading {blob.name}: {e}")
    
    return metrics


def load_metrics_from_dir(dir_path):
    """Load all metrics JSON files from local directory."""
    json_files = glob.glob(os.path.join(dir_path, "*_metrics.json"))
    metrics = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                metric = json.load(f)
                metrics.append(metric)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return metrics


def create_results_table(metrics_list):
    """Create a pandas DataFrame from metrics."""
    rows = []
    for metric in metrics_list:
        row = {
            'app_name': metric.get('app_name', 'unknown'),
            'val_auc': metric.get('metrics', {}).get('val_auc'),
            'val_pr_auc': metric.get('metrics', {}).get('val_pr_auc'),
            'test_auc': metric.get('metrics', {}).get('test_auc'),
            'test_pr_auc': metric.get('metrics', {}).get('test_pr_auc'),
        }
        
        # Add hyperparameters
        hps = metric.get('hyperparameters', {})
        for key in ['max_depth', 'eta', 'subsample', 'colsample_bytree', 
                   'min_child_weight', 'reg_lambda', 'reg_alpha']:
            row[key] = hps.get(key)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('--metrics_path', type=str,
                       help='GCS path to metrics folder (gs://bucket/path/)')
    parser.add_argument('--metrics_dir', type=str,
                       help='Local directory with metrics JSON files')
    parser.add_argument('--output', type=str, default='tuning_results.csv',
                       help='Output CSV file (default: tuning_results.csv)')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Print top N models (default: 10)')
    
    args = parser.parse_args()
    
    if not args.metrics_path and not args.metrics_dir:
        parser.error("Must provide either --metrics_path or --metrics_dir")
    
    # Load metrics
    print("Loading metrics...")
    if args.metrics_path:
        if args.metrics_path.startswith("gs://"):
            metrics = load_metrics_from_gcs(args.metrics_path)
        else:
            metrics = load_metrics_from_dir(args.metrics_path)
    else:
        metrics = load_metrics_from_dir(args.metrics_dir)
    
    if not metrics:
        print("No metrics found!")
        return
    
    print(f"Loaded {len(metrics)} metrics files")
    
    # Create DataFrame
    df = create_results_table(metrics)
    
    # Sort by validation AUC
    df_sorted = df.sort_values('val_auc', ascending=False)
    
    # Save to CSV
    df_sorted.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TUNING RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal models: {len(df)}")
    print(f"Best Validation AUC: {df_sorted.iloc[0]['val_auc']:.4f}")
    print(f"Best Test AUC: {df['test_auc'].max():.4f}")
    print(f"\nTop {args.top_n} models by Validation AUC:")
    print(f"{'='*80}")
    
    # Display top models
    display_cols = ['app_name', 'val_auc', 'test_auc', 'val_pr_auc', 'test_pr_auc',
                   'max_depth', 'eta', 'subsample', 'colsample_bytree', 
                   'min_child_weight', 'reg_lambda']
    display_cols = [c for c in display_cols if c in df_sorted.columns]
    
    print(df_sorted[display_cols].head(args.top_n).to_string(index=False))
    
    # Print best model details
    best = df_sorted.iloc[0]
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best['app_name']}")
    print(f"{'='*80}")
    print(f"Validation AUC: {best['val_auc']:.4f}")
    print(f"Test AUC: {best['test_auc']:.4f}")
    print(f"Validation PR AUC: {best['val_pr_auc']:.4f}")
    print(f"Test PR AUC: {best['test_pr_auc']:.4f}")
    print(f"\nHyperparameters:")
    for col in ['max_depth', 'eta', 'subsample', 'colsample_bytree', 
                'min_child_weight', 'reg_lambda', 'reg_alpha']:
        if col in best and pd.notna(best[col]):
            print(f"  {col}: {best[col]}")


if __name__ == "__main__":
    main()

