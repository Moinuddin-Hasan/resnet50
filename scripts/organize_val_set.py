# resnet50/scripts/organize_val_set.py

import os
import shutil
import sys
import tarfile

# --- Configuration ---
# All paths are hardcoded for simplicity in this one-time script.
ROOT_DIR = "/data/imagenet"
VAL_DIR = os.path.join(ROOT_DIR, "val")
DEVKIT_TAR = os.path.join(ROOT_DIR, "ILSVRC2012_devkit_t12.tar.gz")
DEVKIT_DIR = os.path.join(ROOT_DIR, "ILSVRC2012_devkit_t12")

def main():
    """Main function to organize the ImageNet validation set."""
    print("--- Starting ImageNet Validation Set Organization ---")

    # --- Step 1: Check for Scipy ---
    try:
        import scipy.io as sio
    except ImportError:
        print("\nFATAL ERROR: The 'scipy' module was not found in your Python environment.")
        print("Please install it before running this script.")
        print("Suggestion: Find your PyTorch environment with 'conda env list'")
        print("Then run: /opt/conda/envs/YOUR_ENV_NAME/bin/pip install scipy\n")
        sys.exit(1)

    # --- Step 2: Check for and Unpack the Devkit ---
    if not os.path.exists(DEVKIT_TAR):
        print(f"\nFATAL ERROR: Devkit archive not found at: {DEVKIT_TAR}")
        print("Please download it from image-net.org into the /data/imagenet directory.")
        sys.exit(1)

    if not os.path.exists(DEVKIT_DIR):
        print(f"Unpacking devkit from {DEVKIT_TAR}...")
        with tarfile.open(DEVKIT_TAR, "r:gz") as tar:
            tar.extractall(path=ROOT_DIR)
        print("Devkit unpacked successfully.")

    # --- Step 3: Load Metadata ---
    gt_file = os.path.join(DEVKIT_DIR, "data", "ILSVRC2012_validation_ground_truth.txt")
    meta_file = os.path.join(DEVKIT_DIR, "data", "meta.mat")

    print("Loading metadata from devkit...")
    gt = [int(x.strip()) for x in open(gt_file)]
    meta = sio.loadmat(meta_file)["synsets"]
    
    # Build a mapping from class index (1-1000) to the WordNet ID (e.g., n01440764)
    idx2wnid = {int(m[0][0]): m[0][1][0] for m in meta if int(m[0][0]) <= 1000}
    print("Metadata loaded.")

    # --- Step 4: Move Files ---
    moved_count = 0
    total_files = len(gt)
    print(f"Organizing {total_files} validation images...")

    for i, class_index in enumerate(gt, 1):
        wordnet_id = idx2wnid.get(class_index)
        if not wordnet_id:
            continue

        destination_dir = os.path.join(VAL_DIR, wordnet_id)
        os.makedirs(destination_dir, exist_ok=True)
        
        source_file = os.path.join(VAL_DIR, f"ILSVRC2012_val_{i:08d}.JPEG")
        if os.path.exists(source_file):
            shutil.move(source_file, os.path.join(destination_dir, os.path.basename(source_file)))
            moved_count += 1

    print(f"\n--- Success! Moved {moved_count} of {total_files} images into 1000 class subfolders. ---")

if __name__ == "__main__":
    main()
