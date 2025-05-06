import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import time

def train_model(model, optimizer_cls=torch.optim.Adam, epochs=50, batch_size=128, max_lr=0.001, weight_decay=1e-4, xtrain=None, ytrain=None, device=None):
    
    # Prepare training dataset and loader
    train_ds = torch.utils.data.TensorDataset(xtrain, ytrain)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer and learning rate scheduler
    optimizer = optimizer_cls(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    history = {'Train Loss': [], 'Train Acc': []}
    model.to(device)

    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        total_loss, total_correct = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb.float())
            loss = F.cross_entropy(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            total_correct += (preds == yb).sum().item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        acc = total_correct / len(ytrain)

        history['Train Loss'].append(avg_loss)
        history['Train Acc'].append(acc)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f} - Time: {epoch_time:.2f} seconds")

    total_training_time = time.time() - total_start_time
    print(f"\n Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    
    return pd.DataFrame(history)





