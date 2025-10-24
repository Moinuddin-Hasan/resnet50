#!/bin/bash

# An automated script to start ResNet-50 training and then back up the results to S3.

# Check if an S3 bucket name was provided as an argument.
if [ -z "$1" ]; then
  echo "Error: No S3 bucket name provided."
  echo "Usage: ./run_training.sh s3://your-bucket-name"
  exit 1
fi

S3_BUCKET=$1

# Navigate to the source code directory
cd ../src/

# Activate the correct conda environment
source /opt/conda/bin/activate pytorch

echo "--- Starting ResNet-50 Training ---"

# Run the training script in the background and capture its Process ID (PID).
nohup python train.py --epochs 90 --batch-size 128 > ../outputs/training_console.log 2>&1 &
TRAINING_PID=$! # This special variable gets the PID of the last background command

echo "Training started in the background with PID: $TRAINING_PID"
echo "You can monitor the output with: tail -f ../outputs/training_console.log"

# Wait for the training process to complete.
wait $TRAINING_PID
TRAINING_EXIT_CODE=$? # This gets the exit code (0 for success, non-zero for failure)

echo "--- Training process (PID: $TRAINING_PID) has finished with exit code: $TRAINING_EXIT_CODE ---"

# Check if the training was successful before starting the backup.
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
  echo "Training completed successfully. Proceeding with S3 backup."
  # Go back to the scripts directory to run the backup script
  cd ../scripts/
  ./backup_to_s3.sh "$S3_BUCKET"
else
  echo "Training failed with exit code $TRAINING_EXIT_CODE. Skipping S3 backup."
  echo "Please check the logs in ../outputs/training_console.log for errors."
fi

echo "--- Pipeline Finished ---"
