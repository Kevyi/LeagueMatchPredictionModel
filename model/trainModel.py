import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.model import LoLAttentionWinPredictor
from model.dataset import LoLMatchDataset
from tqdm import tqdm  # optional for a progress bar
from db.database import db
from tqdm import tqdm  # optional for a progress bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LoLAttentionWinPredictor()  

# Loads model if available.
# try:
#     model.load_state_dict(torch.load("champ_predictor.pt", map_location=device))
# finally:
#     pass

model.to(device)

criterion = nn.BCEWithLogitsLoss()      # for binary win/lose prediction
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #changes the learning rate jump throughout epochs.

def train_on_new_matches():
    model.train()                      # put model into “training” mode
    matches = db.getData() #Gets matches, but we should query the ones we got from yesterday/last week.

    print(f"Obtained {len(matches)} matches.")

    datasetLOL = LoLMatchDataset(matches) 
    dataloader = DataLoader(datasetLOL, batch_size = 32, shuffle = True)
    print(f"Training on {len(datasetLOL)} matches.")
    running_loss = 0.0
    overall_running_loss = 0.0
    overall_batch_size = 0

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

    torch.save(model.state_dict(), "champ_predictor.pt")

    print(f"Train Loss: {epoch_loss:.4f}")
    print(f"Overall Training Loss: {overall_running_loss/overall_batch_size:.4f}")


def validate_on_new_matches():
    validationMatches = db.getData(dataType = "validation")
    datasetLOL = LoLMatchDataset(validationMatches)
    dataloader = DataLoader(datasetLOL, batch_size = 32, shuffle = False)

    # — Optional: validation at the end of each epoch —
    model.eval()
    val_loss = 0.0
    running_correct = 0
    with torch.no_grad(): #Basically prevents recording gradient for optimizer.step() as we aren't training the model rn.
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)

            # print(torch.sigmoid(outputs[0]).item()) #we do outputs[0] because in a batch we are training with 32 data vectors.
            val_loss += criterion(outputs.squeeze(), labels).item() * inputs.size(0)

            # 2) Error‐rate accumulation
            probs = torch.sigmoid(outputs.squeeze())              # convert logits → [0,1]
            preds = (probs > 0.5).float()               # threshold at 0.5. I
            running_correct += (preds == labels).sum().item()

    val_loss /= len(dataloader.dataset)
    val_correct_rate = running_correct / len(dataloader.dataset)

    print(f"Val Loss: {val_loss:.4f}")
    print(f"Correct Rate: {val_correct_rate}")
    return val_correct_rate

def predict_win(champion_ids):
    x = torch.tensor([champion_ids], dtype=torch.long)

    model.eval()
    with torch.no_grad(): #Basically prevents recording gradient for optimizer.step() as we aren't training the model rn.
        logits = model(x)              # shape: (1,)
        win_prob = torch.sigmoid(logits).item()  # convert to Python float

    if win_prob > 0.5:
        print(f"Model predicts team 200 (red) will win with {win_prob:.2%} probability.")
    else:
        print(f"Model predicts team 100 (blue) will win with {(1 - win_prob):.2%} probability.")

    #Leaning towards 1 means it's likelier red wins because of the fact that in dataset, red winning = 1.

    return win_prob