#!/bin/bash

# A one-time setup script to download and prepare the ImageNet dataset
# from an S3 bucket onto the EC2 instance's EBS volume.

set -e # Exit immediately if a command exits with a non-zero status.

# --- CONFIGURATION ---
# Check if all required arguments are provided.
if [ "$#" -ne 2 ]; then
    echo "Usage: ./setup_environment.sh s3://your-bucket-name/path-to-tar /path/to/mount/point"
    echo "Example: ./setup_environment.sh s3://my-imagenet-bucket/raw-data /data"
    exit 1
fi

S3_PATH=$1           # Example: s3://my-imagenet-bucket/raw-data
MOUNT_POINT=$2       # Example: /data

# Define local paths
DATASET_DIR="$MOUNT_POINT/imagenet"
TRAIN_TAR_NAME="ILSVRC2012_img_train.tar"
VAL_TAR_NAME="ILSVRC2012_img_val.tar"

echo "--- Starting ImageNet Setup ---"
echo "S3 Source Path: $S3_PATH"
echo "Local Mount Point: $MOUNT_POINT"
echo "Target Dataset Directory: $DATASET_DIR"

# --- EXECUTION ---

# 1. Check if mount point exists
if [ ! -d "$MOUNT_POINT" ]; then
    echo "Error: Mount point '$MOUNT_POINT' does not exist. Please mount your EBS volume first."
    exit 1
fi

# 2. Download data from S3
echo "Downloading training data ($TRAIN_TAR_NAME) from S3..."
aws s3 cp "$S3_PATH/$TRAIN_TAR_NAME" "$MOUNT_POINT/"

echo "Downloading validation data ($VAL_TAR_NAME) from S3..."
aws s3 cp "$S3_PATH/$VAL_TAR_NAME" "$MOUNT_POINT/"
echo "Downloads complete."

# 3. Create the required directory structure
mkdir -p "$DATASET_DIR/train"
mkdir -p "$DATASET_DIR/val"

# 4. Extract the training data
# This is a large file, so it will take a while.
echo "Extracting training data... this will take a long time."
tar -xf "$MOUNT_POINT/$TRAIN_TAR_NAME" -C "$DATASET_DIR/train/"

# The training data is nested in sub-tar files, one for each class.
# We need to extract all of them.
echo "Extracting sub-archives for training data..."
for f in "$DATASET_DIR"/train/*.tar; do
  d=`basename -s .tar $f`
  mkdir -p "$DATASET_DIR/train/$d"
  tar -xf "$f" -C "$DATASET_DIR/train/$d/"
  rm -f "$f" # Remove the sub-tar after extraction
done
echo "Training data extracted."

# 5. Extract and organize the validation data
# The validation data does not come in class folders, so it needs a helper script.
echo "Extracting and organizing validation data..."
tar -xf "$MOUNT_POINT/$VAL_TAR_NAME" -C "$DATASET_DIR/val/"

# Run the helper script to move validation images into class-specific subfolders.
# Make sure the helper script and the val_map.txt file are in the same directory.
./organize_imagenet_val.sh "$DATASET_DIR/val"

echo "Validation data organized."

# 6. Clean up the main tar files
echo "Cleaning up downloaded tar files..."
rm "$MOUNT_POINT/$TRAIN_TAR_NAME"
rm "$MOUNT_POINT/$VAL_TAR_NAME"

echo "--- ImageNet setup complete! ---"
echo "Dataset is ready at $DATASET_DIR"
