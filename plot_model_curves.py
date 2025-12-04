import argparse
import sys
import os
import tempfile
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType


def parse_years(year_list):
    if isinstance(year_list, list):
        return year_list
    elif isinstance(year_list, dict) and 'start' in year_list and 'end' in year_list:
        return list(range(year_list['start'], year_list['end']))
    elif isinstance(year_list, str) and '-' in year_list:
        start, end = map(int, year_list.split('-'))
        return list(range(start, end))
    raise ValueError(f"Invalid year format: {year_list}")


def extract_predictions_to_pandas(spark, predictions_df, sample_fraction=1.0, max_samples=None):
    extract_prob = udf(lambda v: float(v[1]), FloatType())
    
    results_df = predictions_df.select(
        col("presence").alias("true_label"),
        extract_prob("probability").alias("predicted_prob")
    )
    
    total_count = results_df.count()
    if max_samples and total_count > max_samples:
        sample_fraction = max_samples / total_count
        print(f"Sampling {max_samples} rows ({sample_fraction:.4f} fraction) from {total_count} total")
        results_df = results_df.sample(fraction=sample_fraction, seed=42)
    elif sample_fraction < 1.0:
        print(f"Sampling {sample_fraction:.4f} fraction from {total_count} total")
        results_df = results_df.sample(fraction=sample_fraction, seed=42)
    
    pdf = results_df.toPandas()
    y_true = pdf['true_label'].values
    y_prob = pdf['predicted_prob'].values
    
    return y_true, y_prob


def save_plot_to_gcs(plot_func, gcs_path, *plot_args, **plot_kwargs):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        plot_func(*plot_args, output_path=tmp_path, **plot_kwargs)
        
        from google.cloud import storage
        bucket_name = gcs_path.split("/")[2]
        blob_path = "/".join(gcs_path.split("/")[3:])
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(tmp_path)
        os.unlink(tmp_path)
        print(f"Saved plot to: {gcs_path}")
    except Exception as e:
        print(f"Error saving plot to GCS: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def plot_roc_curve(y_true, y_prob, output_path, auc_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve: {output_path}")


def plot_pr_curve(y_true, y_prob, output_path, pr_auc_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    baseline = np.sum(y_true) / len(y_true)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkblue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline (random) = {baseline:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve: {output_path}")


def plot_confusion_matrix_at_threshold(y_true, y_prob, threshold, output_path):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})', fontsize=14, fontweight='bold')
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}'
    plt.text(0.5, -0.15, metrics_text, transform=plt.gca().transAxes,
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def plot_prediction_distribution(y_true, y_prob, output_path):
    plt.figure(figsize=(10, 6))
    
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    
    plt.hist(neg_probs, bins=50, alpha=0.6, label='Negative (True)', color='blue', density=True)
    plt.hist(pos_probs, bins=50, alpha=0.6, label='Positive (True)', color='red', density=True)
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved probability distribution: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model (e.g., gs://bucket/models/model_name)'
    )
    parser.add_argument(
        '--eval_samples',
        type=str,
        required=True,
        help='Path to evaluation samples parquet (e.g., gs://bucket/54m_9neg_1pos)'
    )
    parser.add_argument(
        '--eval_years',
        type=str,
        required=True,
        help='Years to evaluate on (e.g., "2019-2023" or path to config with test_years)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='plots',
        help='Directory to save plots (local path or GCS path like gs://bucket/plots/)'
    )
    parser.add_argument(
        '--save_predictions',
        type=str,
        default=None,
        help='GCS path to save predictions parquet file (optional)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for confusion matrix (default: 0.5)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=1000000,
        help='Maximum number of samples to use for plotting (default: 1000000). Larger datasets will be sampled.'
    )
    parser.add_argument(
        '--app_name',
        type=str,
        default='ModelCurves',
        help='Spark app name'
    )
    
    args = parser.parse_args()
    
    # Create local output dir if not GCS
    is_gcs_output = args.output_dir.startswith("gs://")
    if not is_gcs_output:
        os.makedirs(args.output_dir, exist_ok=True)
    
    spark = (
        SparkSession.builder
        .appName(args.app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.executor.instances", "7")
        .config("spark.executor.cores", "1")
        .config("spark.task.cpus", "1")
        .getOrCreate()
    )
    
    print(f"Loading model from: {args.model_path}")
    try:
        model = PipelineModel.load(args.model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print(f"\nReading evaluation samples from: {args.eval_samples}")
    eval_df = spark.read.parquet(args.eval_samples)
    total_count = eval_df.count()
    print(f"Total samples: {total_count}")
    
    if args.eval_years.endswith('.yaml'):
        try:
            with open(args.eval_years, 'r') as f:
                config = yaml.safe_load(f)
            eval_years = parse_years(config.get('test_years', config.get('val_years', config.get('eval_years'))))
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    else:
        eval_years = parse_years(args.eval_years)
    
    print(f"Filtering to years: {eval_years}")
    eval_df = eval_df.filter(eval_df.year.isin(eval_years))
    eval_count = eval_df.count()
    print(f"Evaluation samples: {eval_count}")
    
    pos_count = eval_df.filter(eval_df.presence == 1).count()
    neg_count = eval_df.filter(eval_df.presence == 0).count()
    print(f"Positive samples: {pos_count}, Negative samples: {neg_count}")
    
    print("\nMaking predictions...")
    predictions = model.transform(eval_df)
    
    print("\nCalculating metrics on full dataset...")
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="presence",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="presence",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )
    
    roc_auc = auc_evaluator.evaluate(predictions)
    pr_auc = pr_evaluator.evaluate(predictions)
    
    print(f"\nMetrics (full dataset):")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC:  {pr_auc:.4f}")
    
    print(f"\nSampling for plotting (max {args.max_samples} samples)...")
    y_true, y_prob = extract_predictions_to_pandas(spark, predictions, max_samples=args.max_samples)
    
    if args.save_predictions:
        print(f"\nSaving predictions to: {args.save_predictions}")
        import pandas as pd
        pred_df = pd.DataFrame({'true_label': y_true, 'predicted_prob': y_prob})
        
        if args.save_predictions.startswith("gs://"):
            pred_spark_df = spark.createDataFrame(pred_df)
            pred_spark_df.write.mode("overwrite").parquet(args.save_predictions)
            print(f"✓ Predictions saved to: {args.save_predictions}")
        else:
            pred_df.to_parquet(args.save_predictions)
            print(f"✓ Predictions saved to: {args.save_predictions}")
    
    print(f"\nGenerating plots...")
    
    if is_gcs_output:
        save_plot_to_gcs(plot_roc_curve, 
                        f"{args.output_dir}/roc_curve.png",
                        y_true, y_prob, auc_score=roc_auc)
        
        save_plot_to_gcs(plot_pr_curve,
                        f"{args.output_dir}/pr_curve.png",
                        y_true, y_prob, pr_auc_score=pr_auc)
        
        save_plot_to_gcs(plot_confusion_matrix_at_threshold,
                        f"{args.output_dir}/confusion_matrix_thresh_{args.threshold:.2f}.png",
                        y_true, y_prob, threshold=args.threshold)
        
        save_plot_to_gcs(plot_prediction_distribution,
                        f"{args.output_dir}/probability_distribution.png",
                        y_true, y_prob)
    else:
        plot_roc_curve(y_true, y_prob, 
                      os.path.join(args.output_dir, 'roc_curve.png'), 
                      roc_auc)
        
        plot_pr_curve(y_true, y_prob,
                     os.path.join(args.output_dir, 'pr_curve.png'),
                     pr_auc)
        
        plot_confusion_matrix_at_threshold(y_true, y_prob, args.threshold,
                                          os.path.join(args.output_dir, f'confusion_matrix_thresh_{args.threshold:.2f}.png'))
        
        plot_prediction_distribution(y_true, y_prob,
                                    os.path.join(args.output_dir, 'probability_distribution.png'))
    
    print(f"\nAll plots saved to: {args.output_dir}")
    
    spark.stop()


if __name__ == "__main__":
    main()

