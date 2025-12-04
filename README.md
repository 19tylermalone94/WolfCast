# WolfCast

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade earthengine-api
earthengine authenticate
```

## Training

```bash
gcloud dataproc jobs submit pyspark \
  gs://wolfcast_training_samples/xgboost_train.py \
  --cluster=cluster-wolf \
  --region=us-east1 \
  -- --config=gs://wolfcast_training_samples/configs/config.yaml
```

## Hyperparameter Tuning

Generate random configs:
```bash
python generate_tuning_configs.py \
  --base_config 54m_9neg_1pos.yaml \
  --num_trials 20 \
  --output_dir tuning_configs/
```

Submit jobs:
```bash
./submit_tuning_jobs.sh
```

Analyze results:
```bash
python analyze_tuning_results.py \
  --metrics_path gs://wolfcast_training_samples/metrics/ \
  --output tuning_results.csv
```

## Evaluation

Test model on different dataset:
```bash
python test_model.py \
  --model_path gs://bucket/models/model_name \
  --eval_samples gs://bucket/54m_9neg_1pos \
  --eval_years 2019-2023
```

Generate plots:
```bash
python plot_model_curves.py \
  --model_path gs://bucket/models/model_name \
  --eval_samples gs://bucket/54m_9neg_1pos \
  --eval_years 2019-2023 \
  --output_dir gs://bucket/plots/
```
