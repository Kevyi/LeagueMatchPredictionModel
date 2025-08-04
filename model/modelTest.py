import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from model import LoLAttentionWinPredictor
from dataset import LoLMatchDataset
from databaseTEMP import db
from tqdm import tqdm  # optional for a progress bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LoLAttentionWinPredictor()  

# # Loads model if available.
# try:
#     model.load_state_dict(torch.load("champ_predictor.pt", map_location=device))
# finally:
#     pass

model.to(device)

criterion = nn.BCEWithLogitsLoss()      # for binary win/lose prediction
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #changes the learning rate jump throughout epochs.

overall_running_loss = 0.0
overall_batch_size = 0

matches = db.getData()

datasetLOL = LoLMatchDataset(matches)
dataloader = DataLoader(datasetLOL, batch_size = 32, shuffle = True)


# 1) Prepare lists
train_losses = []
val_losses   = []   # if you’re also doing a validation pass each epoch



for epoch in range(1):
    model.train()                      # put model into “training” mode
    running_loss = 0.0

    for batch in tqdm(dataloader):
        inputs, labels = batch         # shapes: (B, …), (B,) or (B,1)
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()          # clear previous gradients
        outputs = model(inputs)        # forward pass: shape (B,1)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()                # backpropagate
        optimizer.step()               # update weights

        running_loss += loss.item() * inputs.size(0)
        overall_running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    overall_batch_size += len(dataloader.dataset)
    train_losses.append(epoch_loss)
    print(f"Train Loss: {epoch_loss:.4f}")

print(f"Overall Training Loss: {overall_running_loss/overall_batch_size:.4f}")


validationMatches = db.getData(dataType = "validation")

datasetLOL = LoLMatchDataset(validationMatches)
dataloader = DataLoader(datasetLOL, batch_size = 32, shuffle = False)

# — Optional: validation at the end of each epoch —
model.eval()
val_loss = 0.0
running_incorrect = 0
with torch.no_grad(): #Basically prevents recording gradient for optimizer.step() as we aren't training the model rn.
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        outputs = model(inputs)

        # print(torch.sigmoid(outputs[0]).item()) #we do outputs[0] because in a batch we are training with 32 data vectors.
        val_loss += criterion(outputs.squeeze(), labels).item() * inputs.size(0)

        # 2) Error‐rate accumulation
        probs = torch.sigmoid(outputs.squeeze())              # convert logits → [0,1]
        preds = (probs > 0.5).float()               # threshold at 0.5. I
        running_incorrect += (preds != labels).sum().item()

val_loss /= len(dataloader.dataset)
val_losses.append(val_loss)
val_error_rate = running_incorrect / len(dataloader.dataset)

print(f"           Val   Loss: {val_loss:.4f}")
print(f"Error Loss: {val_error_rate}")


# Visualization if I have epochs.
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/exp1')
# for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses), 1):
#     writer.add_scalar('Loss/Train', t_loss, epoch)
#     writer.add_scalar('Loss/Val',   v_loss, epoch)
# writer.close()



torch.save(model.state_dict(), "champ_predictor.pt")


