# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import argparse
import time
import os
from tqdm import tqdm

# Import from other project files
import config
from dataset import get_dataloaders
from model import create_resnet50
from utils import setup_logging, append_log

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, num_epochs, scaler):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} [T]")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True) # More efficient

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=total_loss/total_samples, acc=f"{(100*correct_predictions/total_samples):.2f}%")

    return total_loss / total_samples, correct_predictions / total_samples

def validate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validating")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=total_loss/total_samples, acc=f"{(100*correct_predictions/total_samples):.2f}%")

    return total_loss / total_samples, correct_predictions / total_samples

def main(args):
    device = torch.device(config.DEVICE)
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    setup_logging(config.LOG_FILE_PATH)
    
    train_loader, val_loader = get_dataloaders(config.DATA_PATH, args.batch_size, config.NUM_WORKERS)
    model = create_resnet50(config.NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    best_val_acc = 0.0
    
    print("--- Starting Training on Full ImageNet ---")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args.epochs, scaler)
        val_loss, val_acc = validate(model, criterion, val_loader, device)
        
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Val Acc: {val_acc:.4f} | LR: {lr:.6f}")
        append_log(config.LOG_FILE_PATH, epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(config.OUTPUT_PATH, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with accuracy: {val_acc:.4f}")
            
    total_time = (time.time() - start_time) / 3600 # hours
    print(f"--- Training Finished in {total_time:.2f} hours ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-50 on ImageNet")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE, help="Initial learning rate")
    
    args = parser.parse_args()
    main(args)