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

# Loads model if available.
try:
    model.load_state_dict(torch.load("champ_predictor.pt", map_location=device))
finally:
    pass

model.eval()

champion_ids = [
    54, 131, 157, 236, 267,   # Blue team (TOP → SUP) #Malp, Diana, Yasuo, Lucian, Nami
    17, 59, 101, 235, 99  # Red team (TOP → SUP) #Aatrox, Jarvan, Xerath, 
]

champion_ids2 = [
    58, 57, 800, 15, 412,   # Blue team (TOP → SUP) #renekton, maokai, mel, sivir, thresh
    266, 104, 268, 110, 53  # Red team (TOP → SUP) #Aatrox, Graves, Azir, Varus, Blitz
]

with open("champions.json", "r") as f:
    champions = json.load(f)
    championDictionary = {value: key for key, value in champions.items()}

teamComp = ["DrMundo", "Skarner", "Anivia", "Jhin", "Nautilus", "Riven", "Nidalee", "Azir", "Zeri", "Pyke"]
teamIds =[]
for item in teamComp:
    if not championDictionary[item]:
        print(f"non existent champion {item}")
        break

    teamIds.append(int(championDictionary[item]))


x = torch.tensor([champion_ids2], dtype=torch.long)  # shape: (1, 10)

with torch.no_grad(): #Basically prevents recording gradient for optimizer.step() as we aren't training the model rn.
    logits = model(x)              # shape: (1,)
    win_prob = torch.sigmoid(logits).item()  # convert to Python float

if win_prob > 0.5:
    print(f"Model predicts team 200 (red) will win with {win_prob:.2%} probability.")
else:
    print(f"Model predicts team 100 (blue) will win with {(1 - win_prob):.2%} probability.")

#Leaning towards 1 means it's likelier red wins because of the fact that in dataset, red winning = 1.




def predict_win(champion_ids, model_path="model.pth"):
    x = torch.tensor([champion_ids], dtype=torch.long)

    model = LoLAttentionWinPredictor()  # or LoLCrossAttentionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        win_prob = torch.sigmoid(logits).item()

    return win_prob
