# src/config.py

# --- Training Hyperparameters ---
DEVICE = "cuda"
NUM_EPOCHS = 90
BATCH_SIZE = 256  # Standard for a V100 GPU like in a p3.2xlarge
LEARNING_RATE = 0.1 # Standard starting LR for BATCH_SIZE=256
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# --- Dataset Information ---
# CRITICAL: This MUST be 1000 for the full ImageNet dataset
NUM_CLASSES = 1000

# Path on the EC2 instance where the mounted EBS volume
# and the ImageNet dataset are located.
DATA_PATH = "/data/imagenet"

# --- Dataloader ---
# Number of parallel workers for loading data.
# 8 or 16 is a good starting point for a p3.2xlarge (8 vCPUs).
NUM_WORKERS = 8

# --- Output ---
# Where to save model checkpoints and logs on the EC2 instance.
OUTPUT_PATH = "outputs"
LOG_FILE_PATH = "outputs/training_log.md"