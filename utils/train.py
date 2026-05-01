import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import PairedContourDataset
from model import ContourGuidedNet
from loss import l1, edge_loss
from utils.config import load_config


def get_device(cfg):
    if cfg["training"]["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["training"]["device"])


def train():
    cfg = load_config()
    device = get_device(cfg)

    ds = PairedContourDataset(
        cfg["data"]["path"],
        cfg["data"]["img_size"]
    )

    dl = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    model = ContourGuidedNet().to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    for epoch in range(cfg["training"]["epochs"]):
        for batch in dl:
            lr = batch["lr"].to(device)
            contour = batch["contour"].to(device)
            hr = batch["hr"].to(device)

            pred = model(lr, contour)

            loss = l1(pred, hr) + 0.1 * edge_loss(pred, hr, device)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch}: {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()
