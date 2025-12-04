#!/bin/bash

CLUSTER=${1:-cluster-wolf}
REGION=${2:-us-east1}
DELAY=${3:-10}
SCRIPT_PATH="gs://wolfcast_training_samples/xgboost_train.py"
CONFIGS_PATH="gs://wolfcast_training_samples/tuning_configs"

echo "Submitting tuning jobs..."
echo "Cluster: $CLUSTER"
echo "Region: $REGION"
echo "Delay between jobs: ${DELAY}s"
echo "Configs path: $CONFIGS_PATH"
echo ""

CONFIG_FILES=$(gsutil ls "${CONFIGS_PATH}/*.yaml" 2>/dev/null)

if [ -z "$CONFIG_FILES" ]; then
    echo "Error: No config files found in $CONFIGS_PATH"
    exit 1
fi

TOTAL=$(echo "$CONFIG_FILES" | wc -l)
echo "Found $TOTAL config files"
echo ""

SUCCESS=0
FAILED=0
COUNT=0

for CONFIG in $CONFIG_FILES; do
    COUNT=$((COUNT + 1))
    CONFIG_NAME=$(basename "$CONFIG")
    
    echo "[$COUNT/$TOTAL] Submitting $CONFIG_NAME..."
    
    gcloud dataproc jobs submit pyspark \
        "$SCRIPT_PATH" \
        --cluster="$CLUSTER" \
        --region="$REGION" \
        -- --config="$CONFIG" \
        > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  [Success] Successfully submitted"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  [Failed] Failed to submit"
        FAILED=$((FAILED + 1))
    fi
    
    if [ $COUNT -lt $TOTAL ] && [ $DELAY -gt 0 ]; then
        echo "  Waiting ${DELAY}s before next submission..."
        sleep $DELAY
    fi
    echo ""
done

echo "=== Summary ==="
echo "Successfully submitted: $SUCCESS"
echo "Failed: $FAILED"
echo "Total: $TOTAL"

