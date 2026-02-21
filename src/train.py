import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Custom modules import
# Fix OpenMP error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.dataset import SIRDataset
from src.models import SIR_LSTM

# --- Hyperparameters (Settings) ---
BATCH_SIZE = 32         # Ek baar mein kitne examples dekhega
LEARNING_RATE = 0.001   # Kitni tezi se seekhega
EPOCHS = 50             # Kitni baar pura data dekhega
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
X_PATH = "data/processed/X_train.npy"
Y_PATH = "data/processed/Y_train.npy"
MODEL_SAVE_PATH = "saved_models/lstm_sir_v1.pth"
PLOT_SAVE_PATH = "plots/training_loss.png"

# Ensure folders exist
os.makedirs("saved_models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def train_model():
    print(f"Using Device: {DEVICE}")
    
    # 1. Load Data
    full_dataset = SIRDataset(X_PATH, Y_PATH)
    
    # Split: 80% Training, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoaders (Ye data ko batches mein feed karte hain)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    model = SIR_LSTM().to(DEVICE)
    
    # 3. Loss Function & Optimizer
    # MSELoss kyun? Kyunki hum numbers predict kar rahe hain (Regression)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Lists to store loss history
    train_losses = []
    val_losses = []
    
    print("Starting Training...")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train() # Set to training mode
        running_loss = 0.0
        
        # Ek-Ek batch uthao
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # A. Forward Pass (Prediction)
            outputs = model(inputs)
            
            # B. Calculate Loss (Galti kitni hui?)
            loss = criterion(outputs, targets)
            
            # C. Backward Pass (Seekho aur Sudharo)
            optimizer.zero_grad() # Purane gradients saaf karo
            loss.backward()       # Galti kahan hui pata karo
            optimizer.step()      # Weights update karo
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 5. Validation Loop (Test without learning)
        model.eval() # Set to evaluation mode
        val_running_loss = 0.0
        with torch.no_grad(): # Validation mein gradients nahi chahiye
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()
                
        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        
    # 6. Save Model
    print("Saving Model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # 7. Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Loss plot saved to {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    train_model()