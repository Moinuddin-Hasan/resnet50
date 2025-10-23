#!/bin/bash

# A script to back up the final model artifacts to a specified S3 bucket.

# Check if an S3 bucket name was provided as an argument.
if [ -z "$1" ]; then
  echo "Error: No S3 bucket name provided."
  echo "Usage: ./backup_to_s3.sh s3://your-bucket-name"
  exit 1
fi

S3_BUCKET=$1
SOURCE_DIR="../outputs"
DEST_DIR="final-artifacts" # A folder within your S3 bucket

# Check if the source directory and files exist.
if [ ! -f "$SOURCE_DIR/best_model.pth" ] || [ ! -f "$SOURCE_DIR/training_log.md" ]; then
  echo "Error: Could not find best_model.pth or training_log.md in $SOURCE_DIR."
  echo "Backup aborted."
  exit 1
fi

echo "--- Starting Backup to S3 ---"
echo "Source Directory: $SOURCE_DIR"
echo "Target S3 Bucket: $S3_BUCKET/$DEST_DIR/"

# Copy the final model and log file to the S3 bucket.
aws s3 cp "$SOURCE_DIR/best_model.pth" "$S3_BUCKET/$DEST_DIR/"
aws s3 cp "$SOURCE_DIR/training_log.md" "$S3_BUCKET/$DEST_DIR/"

# Verify the exit code of the last command to confirm success.
if [ $? -eq 0 ]; then
  echo "--- Backup to S3 completed successfully! ---"
else
  echo "--- Backup to S3 failed. Please check the AWS CLI output above. ---"
  exit 1
fi

exit 0
