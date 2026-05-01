import torch
import torch.nn as nn

l1 = nn.L1Loss()

def edge_loss(pred, target, device):
    sobel_x = torch.tensor(
        [[1,0,-1],[2,0,-2],[1,0,-1]],
        dtype=torch.float32,
        device=device
    ).unsqueeze(0).unsqueeze(0)

    sobel_y = sobel_x.transpose(2, 3)

    def grad(x):
        gx = nn.functional.conv2d(x, sobel_x, padding=1)
        gy = nn.functional.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    return torch.mean(torch.abs(grad(pred) - grad(target)))
