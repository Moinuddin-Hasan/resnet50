# src/utils.py

import os

def setup_logging(log_path):
    """Creates the markdown log file and writes the header."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    header = "| Epoch | Train Loss | Train Acc | Val Loss | Val Acc  | Learning Rate |\n"
    separator = "|-------|------------|-----------|----------|----------|---------------|\n"
    with open(log_path, 'w') as f:
        f.write(header)
        f.write(separator)

def append_log(log_path, epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, lr):
    """Appends a new row to the markdown log file."""
    log_line = f"| {epoch+1}/{num_epochs} | {train_loss:.4f} | {train_acc:.4f} | {val_loss:.4f} | {val_acc:.4f} | {lr:.6f} |\n"
    with open(log_path, 'a') as f:
        f.write(log_line)