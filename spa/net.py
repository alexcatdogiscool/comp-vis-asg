import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import random
import csv
import cv2

DATA_PATH = "/home/alex/Desktop/school/comp-vis/spa/pano-convert/images"
CSV_PATH = "cleaned_data.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SunElevationModel(nn.Module):
    def __init__(self, num_bins=32):
        super().__init__()

        # Load pretrained EfficientNet
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_bins)
        )

    def forward(self, x):
        logits = self.backbone(x)  # shape: (B, 32)
        return logits
    


def logits_to_elevation(logits, num_bins=32):
    probs = torch.softmax(logits, dim=-1)
    bin_centers = torch.linspace(0, 90, steps=num_bins, device=logits.device)
    return (probs * bin_centers).sum(dim=-1)

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint["epoch"]

def soft_label(elevation, num_bins=32, sigma=2.5):
    bin_centers = torch.linspace(0, 90, steps=num_bins)
    probs = torch.exp(-0.5 * ((bin_centers - elevation) / sigma) ** 2)
    probs /= probs.sum()
    return probs.float()

def evaluate(model, loader):
    model.eval()

    total_error = 0
    count = 0

    with torch.no_grad():
        for images, labels, elevations in loader:
            images = images.to(device)

            logits = model(images)
            preds = logits_to_elevation(logits)

            error = torch.abs(preds.cpu() - elevations)

            total_error += error.sum().item()
            count += len(error)

    return total_error / count # mae in degrees


class MyDataset(Dataset):
    def __init__(self, csv_path, num_bins=32):
        self.num_bins = num_bins
        self.data = []

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def elevation_to_bin(self, elevation):
        bin_idx = int((elevation / 90) * self.num_bins)
        return min(bin_idx, self.num_bins - 1)
    
    

    def __getitem__(self, idx):
        row = self.data[idx]

        filename = row['filename']
        elevation = float(row['elevation'])

        img = cv2.imread(f"{DATA_PATH}/{filename}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)# avg values EfficientNet was trained on
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)# give the model what it saw in training

        img = (img - mean) / std

        label = soft_label(elevation, sigma=2.5)

        return img, label, elevation
        

## start the training!!!!

model = SunElevationModel(num_bins=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.KLDivLoss(reduction="batchmean")

full_dataset = MyDataset(CSV_PATH)

val_size = 1000
train_size = len(full_dataset) - val_size
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)



best_val_error = float("inf")

for epoch in range(50):
    model.train()

    total_loss = 0

    for images, labels, elevations in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        log_probs = torch.log_softmax(logits, dim=-1)

        loss = criterion(log_probs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    val_error = evaluate(model, val_loader)

    print(f"Epoch {epoch}")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Val MAE: {val_error:.2f} degrees")

    save_checkpoint(model, optimizer, epoch, "checkpoint_latest.pth")

    if val_error < best_val_error:
        best_val_error = val_error
        save_checkpoint(model, optimizer, epoch, "checkpoint_best.pth")
        print("Saved BEST model")