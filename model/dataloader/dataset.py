from torch.utils.data import Dataset
import torch

class LoLMatchDataset(Dataset):
    def __init__(self, match_list):
        self.samples = []
        for match in match_list:
            b = match["100"]
            r = match["200"]
            winner = match["win"]

            blue = [b["TOP"], b["JUNGLE"], b["MIDDLE"], b["BOTTOM"], b["UTILITY"]]
            red  = [r["TOP"], r["JUNGLE"], r["MIDDLE"], r["BOTTOM"], r["UTILITY"]]
            
            #Gets ordered data and reverse data.

            # Original order
            label = 0 if winner == 100 else 1 #Highlights if blue won, else red won.
            self.samples.append((blue + red, label)) 

            # Reversed order
            flipped_label = 1 - label
            self.samples.append((red + blue, flipped_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        champs, label = self.samples[idx]
        x = torch.tensor(champs, dtype=torch.long) #Long for integer indexing.
        y = torch.tensor(label, dtype=torch.float32) #Float for 0.0-1.0 values.

        # Use long for anything that will go into an Embedding.
        # Use float32 for any values involved in math, especially for loss.

        return x, y
