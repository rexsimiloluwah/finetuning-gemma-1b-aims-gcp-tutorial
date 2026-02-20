#!/bin/bash
# Usage: ./scripts/download_and_upload_gcs.sh [bucket_name]
# If no bucket_name is provided, a default bucket will be used.

# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export PROJECT_ID=$(gcloud config get-value project)

# Default bucket name (change as needed for your course/tutorial)
DEFAULT_BUCKET="dolly15k-bucket-${PROJECT_ID}"

# Use the provided bucket name, otherwise default
BUCKET_NAME=${1:-$DEFAULT_BUCKET}

# Region
REGION="europe-west4"

# Ensure Hugging Face token is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "❌ Please set HUGGINGFACE_TOKEN in your environment or in a .env file"
    exit 1
fi

echo "Using GCS bucket: $BUCKET_NAME"

# Step 1: Download Dolly-15k if not already downloaded
mkdir -p data/raw
if [ ! -f data/raw/dolly.jsonl ]; then
    echo "📥 Downloading Dolly-15k dataset..."
    export HF_HUB_TOKEN=$HUGGINGFACE_TOKEN
    python - <<EOF
import os
from datasets import load_dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
dataset.to_json("data/raw/dolly.jsonl")
EOF
else
    echo "✅ Dolly-15k already exists locally."
fi

# Step 2: Create bucket if it doesn't exist
echo "📤 Creating GCS bucket if needed..."
gsutil mb -l $REGION gs://$BUCKET_NAME || echo "Bucket already exists"

# Step 3: Sync local data to GCS
echo "📤 Syncing local data to gs://$BUCKET_NAME/data/raw/"
gsutil -m rsync -r data/raw gs://$BUCKET_NAME/data/raw/
echo "✅ Dataset upload complete!"