import argparse
import sys
import json
from datetime import datetime

import yaml
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col


def parse_years(year_list):
    if isinstance(year_list, list):
        return year_list
    elif isinstance(year_list, dict) and 'start' in year_list and 'end' in year_list:
        return list(range(year_list['start'], year_list['end']))
    elif isinstance(year_list, str) and '-' in year_list:
        start, end = map(int, year_list.split('-'))
        return list(range(start, end))
    else:
        raise ValueError(f"Invalid year format: {year_list}")


def main():
    parser = argparse.ArgumentParser(
        description='Test a trained model on evaluation dataset'
    )
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
        '--output_metrics',
        type=str,
        default=None,
        help='Path to save metrics JSON (optional)'
    )
    parser.add_argument(
        '--app_name',
        type=str,
        default='ModelValidation',
        help='Spark app name'
    )
    
    args = parser.parse_args()
    
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
    print(f"Class ratio (neg:pos): {neg_count/pos_count:.2f}:1" if pos_count > 0 else "No positive samples")
    
    print("\nMaking predictions...")
    predictions = model.transform(eval_df)
    
    print("\nEvaluating model...")
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
    
    auc = auc_evaluator.evaluate(predictions)
    pr_auc = pr_evaluator.evaluate(predictions)
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"ROC AUC:  {auc:.4f}")
    print(f"PR AUC:   {pr_auc:.4f}")
    print(f"{'='*60}")
    
    metrics = {
        "model_path": args.model_path,
        "eval_samples_path": args.eval_samples,
        "eval_years": eval_years,
        "timestamp": datetime.now().isoformat(),
        "dataset_info": {
            "total_samples": total_count,
            "eval_samples": eval_count,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "class_ratio": float(neg_count / pos_count) if pos_count > 0 else None
        },
        "metrics": {
            "roc_auc": float(auc),
            "pr_auc": float(pr_auc)
        }
    }
    
    if args.output_metrics:
        if args.output_metrics.startswith("gs://"):
            try:
                from google.cloud import storage
                import tempfile
                
                bucket_name = args.output_metrics.split("/")[2]
                blob_path = "/".join(args.output_metrics.split("/")[3:])
                if not blob_path.endswith('.json'):
                    blob_path += ".json"
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    json.dump(metrics, tmp_file, indent=2)
                    tmp_path = tmp_file.name
                
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(tmp_path)
                import os
                os.unlink(tmp_path)
                print(f"\nMetrics saved to: {args.output_metrics}")
            except Exception as e:
                print(f"\nWarning: Failed to save metrics to GCS: {e}")
        else:
            with open(args.output_metrics, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to: {args.output_metrics}")
    
    spark.stop()


if __name__ == "__main__":
    main()

