#!/bin/bash

# A helper script to organize the ImageNet validation set into subfolders.
# This script is required for PyTorch's ImageFolder class to work.
#
# It requires a mapping file to know which image belongs to which class.
# A standard version of this mapping is usually available online.
# You can download it from:
# https://raw.githubusercontent.com/soumith/imagenet-multiGPU.torch/master/valprep.sh
# or create a val_map.txt file with the ground truth labels.

set -e

VAL_DIR=$1
MAP_FILE="val_map.txt" # This file must be in the same directory as this script.

if [ -z "$VAL_DIR" ]; then
    echo "Error: Please provide the path to the validation directory."
    echo "Usage: ./organize_imagenet_val.sh /path/to/imagenet/val"
    exit 1
fi

if [ ! -f "$MAP_FILE" ]; then
    echo "Error: val_map.txt not found. Please download or create it."
    echo "You can often find it by searching for 'imagenet validation ground truth'."
    exit 1
fi

echo "Organizing validation images in: $VAL_DIR"

# Read the mapping file and move images into subdirectories
while read -r line; do
    # Each line is in the format: "image_name.JPEG class_folder"
    # Example: "ILSVRC2012_val_00000001.JPEG n01440764"
    image_name=$(echo $line | awk '{print $1}')
    class_folder=$(echo $line | awk '{print $2}')

    # Create the class directory if it doesn't exist
    mkdir -p "$VAL_DIR/$class_folder"

    # Move the image into its class directory
    if [ -f "$VAL_DIR/$image_name" ]; then
        mv "$VAL_DIR/$image_name" "$VAL_DIR/$class_folder/"
    fi
done < "$MAP_FILE"

echo "Validation set organized."
