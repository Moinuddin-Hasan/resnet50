#!/bin/bash

# A script to start the ResNet-50 training on an EC2 instance.

cd ../src/
source /opt/conda/bin/activate pytorch_latest_p37

echo "Starting ResNet-50 training..."

# Run the training script using nohup to ensure it keeps running
# even if the SSH session is disconnected.
# All console output (stdout and stderr) will be redirected to
# a log file in the 'outputs' directory.
# The '&' at the end runs the process in the background.
nohup python train.py \
    --epochs 90 \
    --batch-size 256 \
    > outputs/training_console.log 2>&1 &

echo "Training started in the background."
echo "You can monitor the output with: tail -f outputs/training_console.log"
echo "You can check the Markdown logs with: cat outputs/training_log.md"
echo "To see the running process, use: ps aux | grep train.py"