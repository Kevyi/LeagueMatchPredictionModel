import torch
import torch.nn as nn
import model.model as model
import model.dataloader.dataset as dataset

from tqdm import tqdm  # optional for a progress bar


def train_one_epoch():
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in tqdm(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)                  # shape: (B,)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

    print(total_loss / len(train_loader.dataset))

    if 90 < validate():
        #save model else do not.
        return

    return

def validate():
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)

    acc = correct / total
    return total_loss / total, acc

# Driver
best_val_loss = float("inf")
for epoch in range(1, 21):
    train_loss = train_one_epoch()
    val_loss, val_acc = validate()
    scheduler.step(val_loss)

    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_lol_model.pt")
