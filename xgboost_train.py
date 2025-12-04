import argparse
import sys
import json
import os
from datetime import datetime

import yaml
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.functions import when, col
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from xgboost.spark import SparkXGBClassifier


def load_config(config_path):
    """Load configuration from YAML file (supports local and GCS paths)."""
    import tempfile
    
    if config_path.startswith("gs://"):
        try:
            from google.cloud import storage
            bucket_name = config_path.split("/")[2]
            blob_path = "/".join(config_path.split("/")[3:])
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.download_to_filename(tmp_path)
            config_path = tmp_path
        except Exception as e:
            print(f"Error: Failed to download config from GCS: {e}")
            sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}")
        sys.exit(1)


def parse_years(year_list):
    """Parse year list from config (handles both lists and range dicts)."""
    if isinstance(year_list, list):
        return year_list
    elif isinstance(year_list, dict) and 'start' in year_list and 'end' in year_list:
        return list(range(year_list['start'], year_list['end']))
    else:
        raise ValueError(f"Invalid year format: {year_list}")

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model for wolf habitat prediction')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file (default: config.yaml)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    APP_NAME = config.get('app_name', 'WolfHabitatTraining_XGB')
    SAMPLES_PATH = config.get('samples_path')
    MODEL_OUTPUT_PATH = config.get('model_output_path')
    TRAIN_YEARS = parse_years(config.get('train_years'))
    VAL_YEARS = parse_years(config.get('val_years'))
    TEST_YEARS = parse_years(config.get('test_years'))
    XGBOOST_CONFIG = config.get('xgboost', {})
    
    if not all([SAMPLES_PATH, MODEL_OUTPUT_PATH, TRAIN_YEARS, VAL_YEARS, TEST_YEARS]):
        print("Error: Missing required config values")
        sys.exit(1)
    
    spark = (
        SparkSession.builder
        .appName(APP_NAME)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.executor.instances", "7")
        .config("spark.executor.cores", "1")
        .config("spark.task.cpus", "1")
        .getOrCreate()
    )
    
    print(f"Reading samples from: {SAMPLES_PATH}")
    df = spark.read.parquet(SAMPLES_PATH)
    print(f"Total samples: {df.count()}")
    
    train_df = df.filter(df.year.isin(TRAIN_YEARS))
    val_df = df.filter(df.year.isin(VAL_YEARS))
    test_df = df.filter(df.year.isin(TEST_YEARS))

    print(f"Train: {train_df.count()}, Val: {val_df.count()}, Test: {test_df.count()}")
    
    pos_count = train_df.filter(train_df.presence == 1).count()
    neg_count = train_df.filter(train_df.presence == 0).count()
    weight_ratio = neg_count / pos_count if pos_count > 0 else 1.0
    
    train_df = train_df.withColumn(
        "weight",
        when(col("presence") == 1, weight_ratio).otherwise(1.0)
    )
    
    numeric_features = [
        "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2",
        "NDVI", "EVI", "NDWI", "NBR", "BSI", "NDSI"
    ]
    
    indexer = StringIndexer(
        inputCol="nlcd_class",
        outputCol="nlcd_indexed",
        handleInvalid="keep"
    )
    
    encoder = OneHotEncoder(
        inputCols=["nlcd_indexed"],
        outputCols=["nlcd_encoded"],
        dropLast=True
    )
    
    feature_cols = numeric_features + ["nlcd_encoded"]
    
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )
    
    xgb = SparkXGBClassifier(
        label_col="presence",
        features_col="features",
        weight_col="weight",
        **XGBOOST_CONFIG
    )
    
    pipeline = Pipeline(stages=[indexer, encoder, assembler, xgb])
    
    print("Training XGBoost model...")
    model = pipeline.fit(train_df)
    
    evaluator = BinaryClassificationEvaluator(
        labelCol="presence",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    val_pred = model.transform(val_df)
    val_auc = evaluator.evaluate(val_pred)
    print(f"Validation AUC: {val_auc:.4f}")
    
    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="presence",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )
    val_pr_auc = pr_evaluator.evaluate(val_pred)
    print(f"Validation PR AUC: {val_pr_auc:.4f}")
    
    test_pred = model.transform(test_df)
    test_auc = evaluator.evaluate(test_pred)
    print(f"Test AUC: {test_auc:.4f}")
    test_pr_auc = pr_evaluator.evaluate(test_pred)
    print(f"Test PR AUC: {test_pr_auc:.4f}")
    
    train_count = train_df.count()
    val_count = val_df.count()
    test_count = test_df.count()
    
    metrics = {
        "config_path": args.config,
        "app_name": APP_NAME,
        "timestamp": datetime.now().isoformat(),
        "model_output_path": MODEL_OUTPUT_PATH,
        "samples_path": SAMPLES_PATH,
        "dataset_info": {
            "train_samples": train_count,
            "val_samples": val_count,
            "test_samples": test_count,
            "train_years": TRAIN_YEARS,
            "val_years": VAL_YEARS,
            "test_years": TEST_YEARS,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "weight_ratio": float(weight_ratio)
        },
        "hyperparameters": XGBOOST_CONFIG,
        "metrics": {
            "val_auc": float(val_auc),
            "val_pr_auc": float(val_pr_auc),
            "test_auc": float(test_auc),
            "test_pr_auc": float(test_pr_auc)
        }
    }
    
    if MODEL_OUTPUT_PATH.startswith("gs://"):
        metrics_path = MODEL_OUTPUT_PATH.replace("/models/", "/metrics/") + "_metrics.json"
        
        try:
            from google.cloud import storage
            import tempfile
            
            bucket_name = metrics_path.split("/")[2]
            blob_path = "/".join(metrics_path.split("/")[3:])
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(metrics, tmp_file, indent=2)
                tmp_path = tmp_file.name
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(tmp_path)
            os.unlink(tmp_path)
            print(f"Metrics saved to: {metrics_path}")
        except Exception as e:
            print(f"Warning: Failed to save metrics to GCS: {e}")
            local_path = f"{APP_NAME}_metrics.json"
            with open(local_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved locally to: {local_path}")
    else:
        metrics_path = os.path.join(MODEL_OUTPUT_PATH, f"{APP_NAME}_metrics.json")
        os.makedirs(os.path.dirname(metrics_path) if os.path.dirname(metrics_path) else '.', exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
    
    model.write().overwrite().save(MODEL_OUTPUT_PATH)
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    
    spark.stop()


if __name__ == "__main__":
    main()
