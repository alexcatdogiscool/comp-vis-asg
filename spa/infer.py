import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import sys

# -------- SETTINGS --------
NUM_BINS = 32
CHECKPOINT_PATH = "checkpoint_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- MODEL --------
class SunElevationModel(nn.Module):
    def __init__(self, num_bins=32):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_bins)
        )

    def forward(self, x):
        return self.backbone(x)


# -------- LOAD CHECKPOINT --------
def load_model(path):
    model = SunElevationModel(NUM_BINS).to(device)

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    return model


# -------- PREPROCESS IMAGE --------
def load_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not load image: {path}")
    
    print(img.shape)
    
    ## crop image so that it is square!
    if img.shape[0] > img.shape[1]:
        img = img[0:img.shape[1], :]
    else:
        img = img[:, 0:img.shape[0]]

    print(img.shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img = img.astype("float32") / 255.0

    # HWC → CHW
    img = torch.tensor(img).permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    img = (img - mean) / std

    # add batch dim
    img = img.unsqueeze(0)

    return img.to(device)


# -------- LOGITS → ELEVATION --------
def logits_to_elevation(logits, num_bins=32):
    probs = torch.softmax(logits, dim=-1)
    bin_centers = torch.linspace(0, 90, steps=num_bins, device=logits.device)
    elevation = (probs * bin_centers).sum(dim=-1)
    return elevation


# -------- MAIN --------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    model = load_model(CHECKPOINT_PATH)
    img = load_image(image_path)

    with torch.no_grad():
        logits = model(img)
        elevation = logits_to_elevation(logits)
        probs = torch.softmax(logits, dim=-1)
        print("Top bin:", torch.argmax(probs).item())
        print("Confidence:", torch.max(probs).item())

    print(f"Predicted sun elevation: {elevation.item():.2f} degrees")