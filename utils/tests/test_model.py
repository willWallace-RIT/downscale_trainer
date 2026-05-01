import torch
from model import ContourGuidedNet

def test_model_forward():
    model = ContourGuidedNet()
    lr = torch.rand(1,3,128,128)
    contour = torch.rand(1,1,128,128)

    out = model(lr, contour)
    assert out.shape == (1,3,128,128)
